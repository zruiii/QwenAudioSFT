from dataclasses import dataclass, field
import json
import math
import logging
import os
import pathlib
import pandas as pd
from typing import Dict, Optional, List, Any

import torch
from torch.utils.data import Dataset

import transformers
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
TS_COMPRESSION_RATIO = 4
TS_PLACEHOLDER_TOKEN = "<|ts_pad|>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


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

        input_id.extend(sys_ids)
        target.extend([IGNORE_TOKEN_ID] * len(sys_ids))

        for msg in source["messages"]:
            role = msg["role"]
            
            if role == "user":
                # 替换模板中的占位符
                # 处理用户输入：将 <news> 换成实际内容，将 <ts> 换成占位符序列
                user_content = msg["content"].replace("<news>", news_content)
                ts_in_placeholder = get_ts_placeholder_str(ts_in)
                user_content = user_content.replace("<ts>", ts_in_placeholder)
                
                full_user_str = f"<|im_start|>user\n{user_content}<|im_end|>\n"
                u_ids = tokenizer.encode(full_user_str)
                input_id.extend(u_ids)
                target.extend([IGNORE_TOKEN_ID] * len(u_ids))

            elif role == "assistant":
                assistant_text = "<|im_start|>assistant\n"
                
                # 只有当 key 存在且内容不为空时，才添加思考文本
                if "reasoning_content" in msg and msg["reasoning_content"]:
                    assistant_text += f"{msg['reasoning_content']}\n"
                
                # 处理输出时序占位符 (ts_out)
                ts_out_placeholder = get_ts_placeholder_str(ts_out)
                assistant_main_content = msg["content"].replace("<ts>", ts_out_placeholder)
                assistant_text += f"{assistant_main_content}<|im_end|>\n"
                
                a_ids = tokenizer.encode(assistant_text)
                
                # 找到角色 Header 的结束位置，Header 部分 (assistant\n) 设为 IGNORE
                header_str = "<|im_start|>assistant\n"
                header_ids = tokenizer.encode(header_str)
                
                input_id.extend(a_ids)
                # 从 Header 之后的内容开始计算 Loss
                target.extend([IGNORE_TOKEN_ID] * len(header_ids) + a_ids[len(header_ids):])

        # Padding 处理
        assert len(input_id) == len(target)
        padding_len = max_len - len(input_id)
        if padding_len > 0:
            input_id.extend([tokenizer.pad_token_id] * padding_len)
            target.extend([IGNORE_TOKEN_ID] * padding_len)

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        ts_infos.append({
            "ts_in": ts_in,
            "ts_out": ts_out
        })

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
            "ts": [ts_in, ts_out],
            "news": news
        }
        
        data_dict = preprocess([source], self.tokenizer, self.max_len)

        return dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            attention_mask=data_dict["attention_mask"][0],
            ts_info=data_dict["ts_infos"][0]
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset
    rank0_print("Loading data...")

    train_data = []
    with open(data_args.data_path, "r") as f:
        for line in f:
            train_data.append(json.loads(line))
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)
    
    if data_args.eval_data_path:
        eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            for line in f:
                eval_data.append(json.loads(line))
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


if __name__ == "__main__":
    model_path = "models/Qwen2.5-0.5B-Instruct" # 请确保路径正确
    
    # 1. 加载 Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    special_tokens = ["<ts_start>", "<ts_end>", TS_PLACEHOLDER_TOKEN]
    num_added_toks = tokenizer.add_tokens(special_tokens, special_tokens=True)
    print(f"Added {num_added_toks} tokens.")
    
    # 注意：如果是训练，还需要 model.resize_token_embeddings(len(tokenizer))
    
    # 加载数据
    ts_data = pd.read_csv("data/v1/ts.csv")
    news_data = pd.read_csv("data/v1/news.csv")
    index_data = pd.read_csv("data/v1/index.csv")

    dataset = SupervisedDataset(
        ts_df=ts_data, 
        news_df=news_data, 
        index_df=index_data, 
        tokenizer=tokenizer, 
        max_news=10, 
        max_len=1024*8
    )

    # 5. 验证第一个样本
    sample = dataset[0]
    print("Input IDs shape:", sample["input_ids"].shape)
    
    # 解码看看模板是否正确
    decoded_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    print("\n--- Decoded Template ---")
    print(decoded_text)
    
    # 验证 Labels 是否正确掩码
    print(sample["labels"].tolist())
    import pdb; pdb.set_trace()