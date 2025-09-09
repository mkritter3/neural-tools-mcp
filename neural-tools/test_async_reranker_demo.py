#!/usr/bin/env python3
"""
Demo script showing async re-ranking performance improvements
Compares immediate blocking vs progressive enhancement approaches
"""

import asyncio
import time
from typing import List, Dict, Any

from src.infrastructure.async_haiku_reranker import (
    AsyncHaikuReRanker, 
    ReRankingMode,
    search_with_progressive_reranking,
    search_with_cached_reranking
)

# Sample search results for demo
SAMPLE_RESULTS = [
    {
        'content': '''
def binary_search(arr, target):
    """Efficient binary search implementation"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
        '''.strip(),
        'score': 0.92,
        'metadata': {'file_path': 'algorithms/binary_search.py', 'complexity': 'O(log n)'}
    },
    {
        'content': '''
async def fetch_user_data(user_id: str):
    """Fetch user data with caching"""
    cache_key = f"user:{user_id}"
    
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        return json.loads(cached_data)
    
    user_data = await database.fetch_user(user_id)
    await redis_client.setex(cache_key, 300, json.dumps(user_data))
    
    return user_data
        '''.strip(),
        'score': 0.87,
        'metadata': {'file_path': 'api/user_service.py', 'pattern': 'caching'}
    },
    {
        'content': '''
class LRUCache:
    """Least Recently Used cache implementation"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.order.append(key)
        '''.strip(),
        'score': 0.83,
        'metadata': {'file_path': 'data_structures/lru_cache.py', 'complexity': 'O(1)'}
    },
    {
        'content': '''
def quicksort(arr):
    """Quick sort implementation"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]  
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
        '''.strip(),
        'score': 0.79,
        'metadata': {'file_path': 'algorithms/quicksort.py', 'complexity': 'O(n log n)'}
    },
    {
        'content': '''
class BloomFilter:
    """Memory-efficient probabilistic data structure"""
    def __init__(self, capacity: int, error_rate: float = 0.1):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array_size = self._calculate_bit_array_size()
        self.hash_count = self._calculate_hash_count()
        self.bit_array = [False] * self.bit_array_size
    
    def add(self, item: str):
        """Add item to the filter"""
        for i in range(self.hash_count):
            index = self._hash(item, i) % self.bit_array_size
            self.bit_array[index] = True
        '''.strip(),
        'score': 0.75,
        'metadata': {'file_path': 'data_structures/bloom_filter.py', 'complexity': 'O(k)'}
    }
]


async def demo_progressive_vs_immediate():
    """Demonstrate progressive vs immediate re-ranking performance"""
    print("🚀 Async Re-ranking Performance Demo")
    print("=" * 50)
    
    query = "efficient algorithm implementation with good time complexity"
    
    # Test 1: Progressive re-ranking (fast initial response)
    print("\n1. Progressive Re-ranking (Fast Initial Response)")
    print("-" * 30)
    
    start_time = time.perf_counter()
    
    async def reranking_callback(result):
        """Callback when re-ranking completes"""
        elapsed = time.perf_counter() - start_time
        print(f"   ✅ Re-ranking completed after {elapsed:.2f}s total")
        print(f"   🎯 Top result after re-ranking: {result.reranked_results[0]['metadata']['file_path']}")
    
    progressive_result = await search_with_progressive_reranking(
        query=query,
        results=SAMPLE_RESULTS,
        context="Looking for high-performance algorithms",
        callback=reranking_callback
    )
    
    initial_response_time = time.perf_counter() - start_time
    print(f"   ⚡ Initial response time: {initial_response_time*1000:.1f}ms")
    print(f"   📊 Initial results: {len(progressive_result.initial_results)} items")
    print(f"   🔄 Re-ranking status: {'Complete' if progressive_result.reranking_complete else 'In Progress'}")
    print(f"   🥇 Top initial result: {progressive_result.initial_results[0]['metadata']['file_path']}")
    
    if progressive_result.estimated_reranking_time:
        print(f"   ⏱️  Estimated re-ranking time: {progressive_result.estimated_reranking_time:.2f}s")
    
    # Wait for re-ranking to complete
    if not progressive_result.reranking_complete:
        print(f"   ⏳ Waiting for re-ranking to complete...")
        
        async_reranker = AsyncHaikuReRanker()
        final_result = await async_reranker.wait_for_reranking(
            progressive_result.reranking_id, 
            timeout=60.0
        )
        
        if final_result and final_result.reranked_results:
            print(f"   ✅ Re-ranking completed!")
            print(f"   🎯 Final top result: {final_result.reranked_results[0]['metadata']['file_path']}")
    
    # Test 2: Immediate re-ranking (blocking)
    print("\n2. Immediate Re-ranking (Blocking)")
    print("-" * 30)
    
    async_reranker = AsyncHaikuReRanker(default_mode=ReRankingMode.IMMEDIATE)
    
    start_time = time.perf_counter()
    immediate_result = await async_reranker.search_and_rerank(
        query=query,
        results=SAMPLE_RESULTS,
        context="Looking for high-performance algorithms"
    )
    total_time = time.perf_counter() - start_time
    
    print(f"   🐌 Total response time: {total_time:.2f}s")
    print(f"   📊 Results: {len(immediate_result.initial_results)} items")
    print(f"   🔄 Re-ranking status: {'Complete' if immediate_result.reranking_complete else 'Failed'}")
    print(f"   🥇 Top result: {immediate_result.initial_results[0]['metadata']['file_path']}")
    
    # Test 3: Cached-only re-ranking (ultra-fast)
    print("\n3. Cached-Only Re-ranking (Ultra-Fast)")
    print("-" * 30)
    
    start_time = time.perf_counter()
    cached_result = await search_with_cached_reranking(
        query=query,
        results=SAMPLE_RESULTS,
        context="Looking for high-performance algorithms"
    )
    cached_time = time.perf_counter() - start_time
    
    print(f"   ⚡ Cache response time: {cached_time*1000:.1f}ms")
    print(f"   📊 Results: {len(cached_result.initial_results)} items")
    print(f"   💾 Used cached results: {'Yes' if cached_result.reranked_results != cached_result.initial_results else 'No'}")
    print(f"   🥇 Top result: {cached_result.initial_results[0]['metadata']['file_path']}")
    
    # Performance summary
    print("\n📈 Performance Summary")
    print("-" * 30)
    print(f"Progressive (initial): ~{initial_response_time*1000:.0f}ms (user sees results immediately)")
    print(f"Immediate (blocking):  ~{total_time*1000:.0f}ms (user waits for re-ranking)")
    print(f"Cached-only:          ~{cached_time*1000:.0f}ms (fastest, if cached)")
    
    speedup = total_time / initial_response_time
    print(f"\n🚀 Progressive is {speedup:.1f}x faster for initial response!")
    print("   User Experience: Vector results → Progressive enhancement → Final re-ranked results")


async def demo_concurrent_rerankings():
    """Demonstrate handling multiple concurrent re-ranking operations"""
    print("\n\n🔄 Concurrent Re-ranking Demo")
    print("=" * 50)
    
    async_reranker = AsyncHaikuReRanker(
        default_mode=ReRankingMode.PROGRESSIVE,
        max_concurrent_rerankings=2
    )
    
    queries = [
        "binary search algorithm implementation",
        "caching mechanism for web applications", 
        "data structure with O(1) operations"
    ]
    
    # Start multiple re-ranking operations
    results = []
    start_time = time.perf_counter()
    
    for i, query in enumerate(queries):
        print(f"Starting query {i+1}: '{query[:30]}...'")
        result = await async_reranker.search_and_rerank(
            query=query,
            results=SAMPLE_RESULTS
        )
        results.append(result)
    
    initial_time = time.perf_counter() - start_time
    print(f"\n⚡ All {len(queries)} initial responses in {initial_time:.2f}s")
    print(f"📊 Average initial response: {(initial_time/len(queries))*1000:.1f}ms")
    
    # Check final stats
    stats = async_reranker.get_stats()
    print(f"\n📈 Final Statistics:")
    print(f"   Active re-rankings: {stats['active_rerankings']}")
    print(f"   Immediate responses: {stats['immediate_responses']}")
    print(f"   Background re-rankings: {stats['background_rerankings']}")
    print(f"   Cache hits: {stats['cache_hits']}")


if __name__ == "__main__":
    # Run the demos
    asyncio.run(demo_progressive_vs_immediate())
    asyncio.run(demo_concurrent_rerankings())
    
    print("\n🎉 Demo completed!")
    print("\nKey Insights:")
    print("• Progressive re-ranking provides immediate results (~100ms)")
    print("• Background re-ranking improves relevance without blocking users")
    print("• Cached results provide ultra-fast responses for repeated queries")
    print("• Users see fast vector results while re-ranking happens behind the scenes")