from dataclasses import dataclass, field
import json
import math
import logging
import os
import pathlib
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any

import torch
from torch.utils.data import Dataset

import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from accelerate.utils import DistributedType
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from build_model import QwenTSConfig, QwenTSForCausalLM

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TS_COMPRESSION_RATIO = 4
TS_PLACEHOLDER_TOKEN = "<|ts_pad|>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    tune_ts_adapter_only: bool = field(
        default=False,
        metadata={"help": "Whether to only train the TS adapter."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the index CSV file."}
    )
    ts_data_path: str = field(
        default=None, metadata={"help": "Path to the time series CSV file."}
    )
    news_data_path: str = field(
        default=None, metadata={"help": "Path to the news CSV file."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    max_news: int = field(
        default=10, 
        metadata={"help": "Maximum number of news items to include in the context."}
    )
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def get_ts_placeholder_str(ts_list):
    """
    根据时序列表长度，生成占位符字符串
    """
    num_tokens = len(ts_list) // TS_COMPRESSION_RATIO
    # 确保至少有一个 token
    num_tokens = max(num_tokens, 1)
    return "<ts_start>" + TS_PLACEHOLDER_TOKEN * num_tokens + "<ts_end>"


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "你是一名资深的时序分析专家，擅长结合文本信息对数值信号进行预测。"
):
    input_ids, targets, ts_infos = [], [], []

    # 系统提示词
    sys_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n"
    sys_ids = tokenizer.encode(sys_prompt)

    for i, source in enumerate(sources):
        input_id, target = [], []
        ts_in, ts_out = source["ts"]
        news_content = source["news"]
        
        # 记录该样本的时序位置信息
        sample_ts_info = {
            "ts_in": ts_in,
            "ts_out": ts_out,
            "in_range": None,
            "out_range": None
        }

        input_id.extend(sys_ids)
        target.extend([IGNORE_TOKEN_ID] * len(sys_ids))

        for msg in source["messages"]:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                content = content.replace("<news>", news_content)
                # 分段处理 <ts>
                parts = content.split("<ts>")
                
                # 编码 <ts> 之前的内容
                u_prefix = f"<|im_start|>user\n{parts[0]}<ts_start>"
                u_prefix_ids = tokenizer.encode(u_prefix)
                input_id.extend(u_prefix_ids)
                target.extend([IGNORE_TOKEN_ID] * len(u_prefix_ids))
                
                # 记录 ts_in 的起始位置
                in_start = len(input_id)
                num_pad = max(len(ts_in) // TS_COMPRESSION_RATIO, 1)
                pad_ids = [tokenizer.convert_tokens_to_ids(TS_PLACEHOLDER_TOKEN)] * num_pad
                input_id.extend(pad_ids)
                target.extend([IGNORE_TOKEN_ID] * len(pad_ids))
                # 记录 ts_in 的结束位置
                in_end = len(input_id)
                sample_ts_info["in_range"] = (in_start, in_end)
                
                # 编码 <ts> 之后的内容
                u_suffix = f"<ts_end>{parts[1]}<|im_end|>\n"
                u_suffix_ids = tokenizer.encode(u_suffix)
                input_id.extend(u_suffix_ids)
                target.extend([IGNORE_TOKEN_ID] * len(u_suffix_ids))

            elif role == "assistant":
                # 同样逻辑处理 assistant 的 <ts>
                assistant_prefix = "<|im_start|>assistant\n"
                if "reasoning_content" in msg and msg["reasoning_content"]:
                    assistant_prefix += f"{msg['reasoning_content']}\n"
                
                parts = content.split("<ts>")
                # 假设 assistant 内容里一定有 <ts>
                a_prefix = f"{assistant_prefix}{parts[0]}<ts_start>"
                a_prefix_ids = tokenizer.encode(a_prefix)
                input_id.extend(a_prefix_ids)
                # Assistant Header 不计入 Loss
                header_ids = tokenizer.encode("<|im_start|>assistant\n")
                target.extend([IGNORE_TOKEN_ID] * len(header_ids) + a_prefix_ids[len(header_ids):])
                
                # 记录 ts_out 的起始位置
                out_start = len(input_id)
                num_pad = max(len(ts_out) // TS_COMPRESSION_RATIO, 1)
                pad_ids = [tokenizer.convert_tokens_to_ids(TS_PLACEHOLDER_TOKEN)] * num_pad
                input_id.extend(pad_ids)
                target.extend(pad_ids) 
                
                out_end = len(input_id)
                sample_ts_info["out_range"] = (out_start, out_end)
                
                a_suffix = f"<ts_end>{parts[1]}<|im_end|>\n"
                a_suffix_ids = tokenizer.encode(a_suffix)
                input_id.extend(a_suffix_ids)
                target.extend(a_suffix_ids)
            
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))

        # 截断与 Padding
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        ts_infos.append(sample_ts_info)

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        ts_infos=ts_infos,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, ts_df, news_df, index_df, tokenizer: transformers.PreTrainedTokenizer, max_news: int, max_len: int):
        super(SupervisedDataset, self).__init__()
        self.ts_df = ts_df
        self.news_df = news_df
        self.index_df = index_df
        self.tokenizer = tokenizer
        self.max_news = max_news
        self.max_len = max_len
        rank0_print("load DataFrame done ...")

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 获取当前行数据
        row = self.index_df.iloc[i]
        ts_start = int(row['ts_start_id'])
        ts_end = int(row['ts_end_id'])
        ts_pred = int(row['ts_pred_id'])
        news_start = int(row['news_start_id'])
        news_end = int(row['news_end_id'])

        ts_in = self.ts_df.iloc[ts_start : ts_end+1]['Close'].tolist()
        ts_out = self.ts_df.iloc[ts_end+1 : ts_pred+1]['Close'].tolist()
        news_in = self.news_df.iloc[news_start: news_end+1]

        # 时序归一化
        ts_in_arr = np.array(ts_in)
        ts_out_arr = np.array(ts_out)
        mu = ts_in_arr.mean()
        sigma = ts_in_arr.std()

        ts_in_norm = ((ts_in_arr - mu) / sigma).tolist()
        ts_out_norm = ((ts_out_arr - mu) / sigma).tolist()
        
        # 处理新闻
        dates = news_in['date'].tolist()[:self.max_news]
        titles = news_in['title'].tolist()[:self.max_news]
        summary = news_in['summary'].tolist()[:self.max_news]
        
        news = ""
        for i in range(len(dates)-1, -1, -1):
            date_str = dates[i][:10]            # 日期格式固定，只选择年月日
            title_str = titles[i]
            summary_str = summary[i]

            news += f"- {date_str}: {title_str}\n{summary_str}\n"

        source = {
            "messages": [
                {
                    "role": "user",
                    "content": "请你根据历史新闻:\n<news>\n以及历史时序值:\n<ts>\n预测未来的时序"
                },
                {
                    "role": "assistant",
                    "content": "<ts>"
                }
            ],
            "ts": [ts_in_norm, ts_out_norm],
            "news": news
        }
        
        data_dict = preprocess([source], self.tokenizer, self.max_len)

        return dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            attention_mask=data_dict["attention_mask"][0],
            ts_infos=data_dict["ts_infos"][0]
        )

class CustomDataCollator(transformers.DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        ts_infos_list = [feature.pop('ts_infos') for feature in features]
        batch = super().__call__(features)

        batch['ts_infos'] = {
            "ts_in": torch.tensor([x['ts_in'] for x in ts_infos_list], dtype=torch.float),
            "ts_out": torch.tensor([x['ts_out'] for x in ts_infos_list], dtype=torch.float),
            "in_range": torch.tensor([x['in_range'] for x in ts_infos_list], dtype=torch.long),   # (B, 2)
            "out_range": torch.tensor([x['out_range'] for x in ts_infos_list], dtype=torch.long) # (B, 2)
        }

        return batch
    
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")

    ts_df = pd.read_csv(data_args.ts_data_path)
    news_df = pd.read_csv(data_args.news_data_path, low_memory=False) # 避免 DtypeWarning
    index_df = pd.read_csv(data_args.data_path)

    train_dataset = dataset_cls(
        ts_df=ts_df,
        news_df=news_df,
        index_df=index_df,
        tokenizer=tokenizer,
        max_len=max_len,
        max_news=data_args.max_news
    )

    if data_args.eval_data_path:
        eval_index_df = pd.read_csv(data_args.eval_data_path)
        eval_dataset = dataset_cls(
            ts_df=ts_df,
            news_df=news_df,
            index_df=eval_index_df,
            tokenizer=tokenizer,
            max_len=max_len,
            max_news=data_args.max_news
        )
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    # *********** Set Configuration ***********
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (model_args, data_args, training_args, lora_args) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank
    
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )
    
    # *********** Load Model & Tokenizer ***********
    config = QwenTSConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # 加载自定义的 QwenTS 模型
    model = QwenTSForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
    )

    if model_args.tune_ts_adapter_only:
        rank0_print("Special Mode: Freeze LLM backbone and VAE, only training TS adapter.")
        
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if "ts_proj" in name:
                param.requires_grad = True
                rank0_print(f"Unfreezing parameter: {name}")

    # 加载 Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 4. 冻结与 LoRA 设置
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            # 如果扩展了词表，通常建议将 embed_tokens 和 lm_head 加入 modules_to_save
            modules_to_save=["ts_proj", "embed_tokens", "lm_head"] 
        )
        model = get_peft_model(model, lora_config)
        
        # 确保 ts_adapter 相关层在 LoRA 模式下依然可训练
        for name, param in model.named_parameters():
            if "ts_proj" in name or "ts_encoder" in name:
                param.requires_grad = True

        model.print_trainable_parameters()
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # *********** Load Dataset & Trainer ***********
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    data_collator = CustomDataCollator(tokenizer=tokenizer)
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        data_collator=data_collator,
        **data_module
    )

    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)

if __name__ == "__main__":
    train()
    # from build_model import QwenTSConfig, QwenTSForCausalLM

    # def test_model_loading(model_path: str):
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
        
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    #     config = QwenTSConfig.from_pretrained(model_path)
    #     print(f"Config 中的词表大小: {config.vocab_size}")
    #     print(f"Tokenizer 中的实际词表大小: {len(tokenizer)}")
    
    #     model = QwenTSForCausalLM.from_pretrained(
    #         model_path,
    #         config=config, # 传入显式修正后的 config
    #         device_map=device,
    #         torch_dtype=torch.float16 if device == "cuda" else torch.float32
    #     )
    #     print("✅ 加载成功！")

    #     return model, tokenizer

    # model, tokenizer = test_model_loading("save/v1")
    # import pdb; pdb.set_trace()
    

    
    # # 加载数据
    # ts_data = pd.read_csv("data/v1/ts.csv")
    # news_data = pd.read_csv("data/v1/news.csv")
    # index_data = pd.read_csv("data/v1/index.csv")

    # dataset = SupervisedDataset(
    #     ts_df=ts_data, 
    #     news_df=news_data, 
    #     index_df=index_data, 
    #     tokenizer=tokenizer, 
    #     max_news=1, 
    #     max_len=1024*4
    # )

    # data_collator = CustomDataCollator(tokenizer=tokenizer)

    # # 2. 构建 DataLoader
    # from torch.utils.data import DataLoader
    # test_loader = DataLoader(
    #     dataset, 
    #     batch_size=8, 
    #     shuffle=False, 
    #     collate_fn=data_collator
    # )

    # # 3. 取出一个 Batch
    # print(f"--- 正在提取第一个 Batch (Size: {8}) ---")
    # batch = next(iter(test_loader))

    # # 4. 检查 Batch 的结构和形状
    # print(f"Input IDs shape: {batch['input_ids'].shape}")   # 应该是 [B, Max_Len_in_Batch]
    # print(f"Labels shape: {batch['labels'].shape}")
    
    # # 验证时序数据是否已成功 Batch 化
    # ts_infos = batch['ts_infos']
    # print(f"Batch TS-In shape: {ts_infos['ts_in'].shape}")   # 应该是 [B, L_in]
    # print(f"Batch TS-Out shape: {ts_infos['ts_out'].shape}") # 应该是 [B, L_out]
    # print(f"In-Range tensor:\n{ts_infos['in_range']}")      # 应该是 [B, 2]

    # # 5. 将数据移动到设备上
    # # 文本部分
    # input_ids = batch['input_ids'].to("cuda")
    # labels = batch['labels'].to("cuda")
    # attention_mask = batch['attention_mask'].to("cuda")
    
    # # 时序部分 (由于是字典，我们需要手动移动其内部 Tensor)
    # ts_infos_device = {k: v.to("cuda") for k, v in ts_infos.items()}

    # # 6. 前向传播测试
    # print("--- 正在进行 Forward 测试 ---")
    # try:
    #     with torch.no_grad():
    #         outputs = model(
    #             input_ids=input_ids,
    #             labels=labels,
    #             attention_mask=attention_mask,
    #             ts_infos=ts_infos_device,
    #             return_dict=True
    #         )
        
    #     print("✅ Batch Forward 成功！")
    #     print(f"Loss: {outputs.loss.item():.4f}" if outputs.loss is not None else "Loss 为空 (检查 Labels 是否正确)")
    #     print(f"Logits shape: {outputs.logits.shape}")

    # except Exception as e:
    #     print(f"❌ Batch Forward 失败: {str(e)}")
    #     import traceback
    #     traceback.print_exc()

    
    # # 解码看看模板是否正确
    # decoded_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    # print("\n--- Decoded Template ---")
    # print(decoded_text)
    
    # # 验证 Labels 是否正确掩码
    # print(sample["labels"].tolist())
    # import pdb; pdb.set_trace()