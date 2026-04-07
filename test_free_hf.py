"""Test FREE HuggingFace Serverless models."""
import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

# HF Serverless API (FREE, no credits needed)
# Uses different endpoint than Inference Providers
API_BASE = "https://api-inference.huggingface.co/v1"

# Free models to test (Serverless API)
FREE_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",     # 3.8B - Fast
    "meta-llama/Llama-3.2-3B-Instruct",     # 3B - Smaller but free
    "google/gemma-2-9b-it",                  # 9B - Good quality
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # 1.7B - Very fast
]

async def test_model(model_name):
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE)
    
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say 'HF Serverless works!' in 3 words."},
                ],
                max_tokens=20,
                temperature=0.1,
            ),
            timeout=30
        )
        
        reply = response.choices[0].message.content
        print(f"✅ SUCCESS! Model replied: {reply}")
        print(f"   Model: {response.model}")
        return True
        
    except asyncio.TimeoutError:
        print(f"⏳ TIMEOUT (model loading, try again in 30s)")
        return False
    except Exception as e:
        error_msg = str(e)
        if "402" in error_msg:
            print(f"❌ PAID MODEL (need credits)")
        elif "404" in error_msg:
            print(f"❌ NOT FOUND")
        elif "503" in error_msg:
            print(f"⏳ LOADING (model is cold, try again)")
        else:
            print(f"❌ FAILED: {error_msg[:100]}")
        return False

async def main():
    print(f"Token: {'Found' if HF_TOKEN else 'NOT FOUND'}")
    print(f"Endpoint: {API_BASE}")
    print(f"\nTesting FREE HuggingFace Serverless models...\n")
    
    results = {}
    for model in FREE_MODELS:
        results[model] = await test_model(model)
        await asyncio.sleep(1)  # Rate limit
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model, success in results.items():
        status = "✅ WORKS" if success else "❌ Failed"
        print(f"{status}: {model}")

asyncio.run(main())
