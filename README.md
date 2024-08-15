# QwenAudioSFT
The repoduction codes for Qwen-Audio Fine-tuning

Since the official [Qwen-Audio repository](https://github.com/QwenLM/Qwen-Audio) doesn't provide a fine-tuning script, I've recreated the fine-tuning code for the Qwen-Audio model here, inspired by the fine-tuning scripts for the previous Qwen series. Below are some steps to help you get started:

### Step 1: Install Dependencies
First, install the necessary packages by running:
```bash
pip install -r requirements.txt
```

### Step 2: Replace the Original `modeling_qwen.py` File
I'm using the [Qwen-Audio-Chat](https://huggingface.co/Qwen/Qwen-Audio-Chat) model for this. Replace the `modeling_qwen.py` file in the original model with the version from this repository. This allows the AudioEncoder module to remain frozen during fine-tuning.

### Step3: Prepare Your Data
Please format your training data similar to the example below:
```
sample = {
    "messages": [
        {
            "role": "user",
            "audio": "data/audio/T0055G0007S0001.wav",
            "content": "Please translate this audio into Chinese."
        },
        {
            "role": "assistant",
            "content": "没有人知道他为什么要这么做。"
        }
    ]
}
```

### Step 4: Start Training
Next, update the configurations in `finetune.sh` just like you would when fine-tuning the previous models in the Qwen series. Then, kick off the training script with:
```bash
sh finetune.sh
```




