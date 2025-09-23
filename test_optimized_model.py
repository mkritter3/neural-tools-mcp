#!/usr/bin/env python3
"""
Quick test of the optimized model performance
"""

import asyncio
import aiohttp
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_optimized_model():
    """Test direct Ollama performance with optimized model"""

    logger.info("🔄 Testing graphiti-coder-optimized model performance...")

    try:
        async with aiohttp.ClientSession() as session:
            # Test payload - simple entity extraction task
            test_payload = {
                "model": "graphiti-coder-optimized",
                "messages": [
                    {
                        "role": "user",
                        "content": "Extract entities from this code: def test_function(): return 'hello'"
                    }
                ],
                "stream": False
            }

            start_time = time.time()

            async with session.post(
                "http://localhost:11434/v1/chat/completions",
                json=test_payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time

                    logger.info(f"✅ Model responded in {elapsed:.2f}s")
                    logger.info(f"✅ Response: {result['choices'][0]['message']['content'][:100]}...")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"❌ HTTP {response.status}: {error_text}")
                    return False

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Error after {elapsed:.2f}s: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_optimized_model())
    print(f"\n{'='*40}")
    if success:
        print("🎉 Optimized model working!")
    else:
        print("❌ Performance issues remain")
    print(f"{'='*40}")