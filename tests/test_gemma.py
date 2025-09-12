#!/usr/bin/env python3
"""Test Gemma metadata tagging"""

import asyncio
import json
import aiohttp


async def test_gemma():
    """Test Gemma LLM for code metadata extraction"""
    
    test_code = """
def calculate_total(items):
    '''Calculate total price of items'''
    total = 0
    for item in items:
        total += item['price'] * item['quantity']
    return total
"""
    
    # Simpler prompt without format=json (which is slow)
    prompt = f"""Return only JSON for this code:
{test_code}

{{"status": "active", "component_type": "utility", "complexity_score": 0.3, "dependencies": []}}"""

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:48001/api/generate",
                json={
                    "model": "qwen3:0.6b",
                    "prompt": prompt,
                    # Don't use format="json" - it's 10x slower
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.7,
                        "top_k": 10,
                        "num_predict": 150
                    }
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Gemma responded successfully!")
                    print("\nRaw response:")
                    print(json.dumps(result, indent=2))
                    
                    # Try to parse the response
                    try:
                        metadata = json.loads(result.get('response', '{}'))
                        print("\nExtracted metadata:")
                        print(json.dumps(metadata, indent=2))
                    except json.JSONDecodeError as e:
                        print(f"\n⚠️ Response not valid JSON: {e}")
                        print(f"Response text: {result.get('response', 'No response')}")
                else:
                    print(f"❌ Gemma returned status {response.status}")
                    
    except Exception as e:
        import traceback
        print(f"❌ Error testing Gemma: {e}")
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    print("Testing Gemma 2B metadata extraction...")
    asyncio.run(test_gemma())