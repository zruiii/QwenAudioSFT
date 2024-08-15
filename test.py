import json

data = {
    "messages": [
        {
            "role": "user",
            "audio": "data/T0055G0007S0001.wav",
            "content": "Please translate this audio into Chinese."
        },
        {
            "role": "assistant",
            "content": "没有人知道他为什么要这么做。"
        }
    ]
}

with open("data/toy_data.jsonl", "w") as f:
    f.write(json.dumps(data, ensure_ascii=False))