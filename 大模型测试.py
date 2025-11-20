#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式调用公司大模型网关示例
POST  /chat/completions  HTTP/1.1
Host: ai.caijj.net
Authorization: Bearer <your_token>
Content-Type: application/json
{"stream":true,"model":"pri_deepseek-r1-671B_shuhe-tencent","messages":[...]}
"""

import json
import sys
import requests

# 1. 基本配置
GATEWAY = "http://ai.caijj.net/chat/completions"
TOKEN   = "sk-6X8QyIvh8MIYHPY3poaa_A"          # 替换成你的个人 Token
MODEL   = "pub_qwen3-coder-plus_bailian"  # 也可换成其它可用模型

# 2. 组装请求体
payload = {
    "stream": True,
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "请用 100 字以内介绍 Python 的生成器"}
    ]
}

# 3. 发送流式请求
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

try:
    resp = requests.post(GATEWAY, headers=headers, json=payload, stream=True, timeout=60)
    resp.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"[ERROR] 请求失败: {e}")
    sys.exit(1)

# 4. 逐行解析 SSE 流式输出
print("Assistant: ", end="", flush=True)
for line in resp.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data:"):
        continue
    data = line[5:].strip()
    if data == "[DONE]":          # 流结束标志
        break
    try:
        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
    except (json.JSONDecodeError, KeyError):
        continue
print()  # 最后换行