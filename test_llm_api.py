import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not api_key:
    print("ERROR: OPENAI_API_KEY is missing.")
    sys.exit(1)

client_kwargs = {"api_key": api_key}
if base_url:
    client_kwargs["base_url"] = base_url

client = OpenAI(**client_kwargs)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "你是一个简洁的中文助手。",
            },
            {
                "role": "user",
                "content": "请用一句话说明 RAG 系统为什么需要引用来源。",
            },
        ],
        temperature=0.2,
    )

    print("SUCCESS: LLM API 可用")
    print(response.choices[0].message.content)

except Exception as e:
    print("ERROR: LLM API 调用失败")
    print(type(e).__name__)
    print(e)
    sys.exit(1)