import os, json
from typing import Optional, List, Union, Dict, Tuple

from transformers import Qwen2Config
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import torch
import torch.nn as nn
from vae import vqvae as VQVAE

class QwenTSConfig(Qwen2Config):
    model_type = "qwen_ts"

    def __init__(
        self,
        vae_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vae_config = vae_config
        self.ts_pad_token_id = 151667

class QwenTSForCausalLM(Qwen2ForCausalLM):
    config_class = QwenTSConfig

    def __init__(self, config):
        # 1. 调用父类初始化，会自动创建 self.model 和 self.lm_head
        super().__init__(config) 
        
        # 2. 仅添加时序相关的特殊组件
        if config.vae_config is not None:
            full_vae = VQVAE(config.vae_config)
            self.ts_encoder = full_vae.encoder
            self.ts_vq = full_vae.vq
            
            # 投影层：将 VAE 的 128 维映射到 LLM 的 896 维
            self.ts_proj = nn.Linear(
                config.vae_config['embedding_dim'], 
                config.hidden_size
            )
        else:
            self.ts_encoder = self.ts_vq = self.ts_proj = None

        # 初始化新增参数权重
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        ts_infos: Dict[str, torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # 增加这个参数
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # 1. 构造 inputs_embeds
        if inputs_embeds is None:
            # 此时 self.model 是 Transformer 核心 (Qwen2Model)
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # 2. 处理时序信息并替换 inputs_embeds
        if ts_infos is not None:
            # --- 批量处理输入时序 ts_in ---
            ts_in = ts_infos['ts_in'].to(device=device, dtype=dtype).unsqueeze(1)
            with torch.no_grad():
                # 编码并投影
                z_in = self.ts_encoder(ts_in, self.config.vae_config['compression_factor']).transpose(1, 2)
                ts_in_feats = self.ts_proj(z_in) # (B, L_latent, 896)

            # 替换占位符
            in_ranges = ts_infos['in_range']
            for i in range(inputs_embeds.shape[0]):
                s, e = in_ranges[i]
                actual_len = min(e - s, ts_in_feats.shape[1])
                inputs_embeds[i, s : s + actual_len] = ts_in_feats[i, :actual_len]

            # --- 批量处理输出标签 ts_out ---
            if labels is not None:
                ts_out = ts_infos['ts_out'].to(device=device, dtype=dtype).unsqueeze(1)
                with torch.no_grad():
                    z_out = self.ts_encoder(ts_out, self.config.vae_config['compression_factor'])
                    _, _, _, _, encoding_indices, _ = self.ts_vq(z_out)
                    encoding_indices = encoding_indices.view(ts_out.shape[0], -1)
                    # import pdb; pdb.set_trace()
                
                vq_offset = self.config.vocab_size - 256
                out_ranges = ts_infos['out_range']
                for i in range(labels.shape[0]):
                    s, e = out_ranges[i]
                    actual_out_len = min(e - s, encoding_indices.shape[1])
                    labels[i, s : s + actual_out_len] = encoding_indices[i, :actual_out_len].long() + vq_offset

        # 3. 直接调用父类的 forward
        return super().forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

if __name__ == "__main__":
    def merge_and_save_new_model(qwen_dir, vae_path, output_dir):
        # 1. 加载 Tokenizer 并添加新词
        tokenizer = AutoTokenizer.from_pretrained(qwen_dir)
        # 添加结构化特殊词
        special_tokens = ["<|ts_start|>", "<|ts_end|>", "<|ts_pad|>"]
        # 添加 VQ 离散词 (ts_code_0 到 ts_code_255)
        ts_code_tokens = [f"<ts_code_{i}>" for i in range(256)]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens + ts_code_tokens})

        # 2. 读取 VAE 配置
        vae_dir = os.path.dirname(vae_path)
        vqvae_config_path = os.path.join(vae_dir, '..', 'configs', 'config_file.json')
        with open(vqvae_config_path, 'r') as f:
            full_config = json.load(f)
        vae_params = full_config['vqvae_config']

        # 3. 创建并调整模型
        qwen_config = Qwen2Config.from_pretrained(qwen_dir)
        new_config = QwenTSConfig(
            **qwen_config.to_dict(),
            vae_config=vae_params
        )
        
        # 实例化 QwenTS 模型
        model = QwenTSForCausalLM(new_config)
        
        # 加载原始 Qwen 权重
        qwen_base = Qwen2ForCausalLM.from_pretrained(qwen_dir)
        model.load_state_dict(qwen_base.state_dict(), strict=False)
        
        # 这里不需要调整原始 Qwen 的 Embedding/Project 层，因为它相对词表本身就有缓冲区 (151936 vs 151665)
        with torch.no_grad():
            # 原始词表大小
            start_id = 151665
            std = new_config.initializer_range
            model.get_input_embeddings().weight[start_id:len(tokenizer)].normal_(mean=0, std=std)
            model.get_output_embeddings().weight[start_id:len(tokenizer)].normal_(mean=0, std=std)
        
        # 4. 加载 VAE 权重
        vae_state_dict = torch.load(vae_path, map_location='cpu')
        model.ts_encoder.load_state_dict({k.replace('encoder.', ''): v for k,v in vae_state_dict.items() if k.startswith('encoder.')})
        model.ts_vq.load_state_dict({k.replace('vq.', ''): v for k,v in vae_state_dict.items() if k.startswith('vq.')})
        
        # 5. 保存模型和 Tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"模型已保存，词表大小由 {qwen_config.vocab_size} 扩展为 {len(tokenizer)}")
        import pdb; pdb.set_trace()
    
    merge_and_save_new_model(
        qwen_dir = "models/Qwen2.5-0.5B-Instruct",
        vae_path = "../multimodalTS-main/save/vae/stock5chan_256x128code_useinputnorm/CD128_CW256_CF4_BS8192_ITR200000_lr0.0005/checkpoints/model_epoch_14.pth",
        output_dir = "save/v1"
    )