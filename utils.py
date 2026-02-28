PROMPT_TEMPLATE = """历史新闻:
<news>
历史时序:
<ts>
请你结合历史新闻以及历史时序，对未来的时序进行预测。
"""



message_dict = {
        "messages": [
            {
                "role": "user",
                "content": "请你根据历史新闻:\n<news>\n以及历史序列:\n" + len(var)*"<var><ts>" + "预测未来7天的值"
            },
            {
                "role": "assistant",
                "reasoning_content": "",
                "content": "<ts>"
            }
        ],
        "ts": [
            [100] * 3,
            [100, 100, 100],
            [100, 200, 300],
            [400, 500, 600]
        ],
        "var": [
            "OPEN",
            "OPEN",
            "CLOSE"
        ],
        "news": [
            "第一条新闻",
            "第二条新闻"
        ]
    }