#!/usr/bin/env python3
"""
Test Redis cache integration specifically
"""

import asyncio
import time
import sys
sys.path.append('neural-tools/src')

async def test_cache():
    from servers.services.nomic_service import NomicService

    print("Testing Redis Cache...")
    nomic = NomicService()
    await nomic.initialize()

    # Test with 3 different texts
    test_texts = [
        "test cache one",
        "test cache two",
        "test cache three"
    ]

    for text in test_texts:
        print(f"\n📝 Testing: '{text}'")

        # First call - miss
        start = time.time()
        emb1 = await nomic.get_embedding(text)
        miss_time = time.time() - start

        # Second call - hit
        start = time.time()
        emb2 = await nomic.get_embedding(text)
        hit_time = time.time() - start

        speedup = miss_time / hit_time if hit_time > 0 else 0
        print(f"   Miss: {miss_time:.3f}s, Hit: {hit_time:.3f}s, Speedup: {speedup:.0f}x")

        if speedup > 100:
            print("   ✅ Cache working!")

if __name__ == "__main__":
    asyncio.run(test_cache())