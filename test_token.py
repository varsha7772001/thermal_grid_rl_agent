"""Test if HF_TOKEN is working with the API."""
import os
from openai import AsyncOpenAI
import asyncio
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("API_BASE_URL", os.environ.get("OPENAI_API_BASE", "https://router.huggingface.co/v1"))
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

print(f"Token found: {'YES' if HF_TOKEN else 'NO'}")
print(f"API Base: {API_BASE}")
print(f"Model: {MODEL_NAME}")
print()

if not HF_TOKEN:
    print("❌ No token found in environment variables.")
    exit(1)

async def test_api():
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE)
    
    try:
        print("Testing API connection...")
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API is working!' in one sentence."},
            ],
            max_tokens=50,
            temperature=0.1,
        )
        
        reply = response.choices[0].message.content
        print(f"✅ SUCCESS! Model replied: {reply}")
        print(f"   Model used: {response.model}")
        print(f"   Tokens used: {response.usage}")
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")

asyncio.run(test_api())
