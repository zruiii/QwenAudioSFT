import torch
from transformers import AutoTokenizer
from build_model import QwenTSConfig, QwenTSForCausalLM 

from transformers import AutoConfig

def test_model_loading(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 先加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. 关键：先加载 Config，并观察它的词表大小
    config = QwenTSConfig.from_pretrained(model_path)
    print(f"Config 中的词表大小: {config.vocab_size}")
    print(f"Tokenizer 中的实际词表大小: {len(tokenizer)}")

    # 3. 如果两者不一致，以 Tokenizer 为准 (或者直接设为报错要求的 151936)
    # 建议直接设为报错信息里 state_dict 的大小：151936
    config.vocab_size = 151936 

    # 4. 传入修改后的 config 进行加载
    model = QwenTSForCausalLM.from_pretrained(
        model_path,
        config=config, # 传入显式修正后的 config
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    print("✅ 加载成功！")

if __name__ == "__main__":
    test_model_loading("save/v1")