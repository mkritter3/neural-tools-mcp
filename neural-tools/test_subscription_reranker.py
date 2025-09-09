#!/usr/bin/env python3
"""
Test script for subscription-based Haiku reranker
Tests using your Claude subscription without separate API key
"""

import asyncio
import time
from src.infrastructure.subscription_haiku_reranker import (
    SubscriptionHaikuReRanker, rerank_with_subscription
)

async def test_subscription_reranker():
    """Test the subscription-based reranker"""
    print("🧪 Testing Subscription-Based Haiku Reranker")
    print("=" * 50)
    
    # Sample search results for testing
    sample_results = [
        {
            'content': '''
def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using dynamic programming"""
    if n <= 1:
        return n
    
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
    
    return curr
            '''.strip(),
            'score': 0.85,
            'metadata': {'file_path': 'algorithms/fibonacci.py', 'language': 'python'}
        },
        {
            'content': '''
async def fetch_data_parallel(urls: List[str]) -> List[Dict]:
    """Fetch data from multiple URLs in parallel"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
            '''.strip(),
            'score': 0.78,
            'metadata': {'file_path': 'networking/async_client.py', 'language': 'python'}
        },
        {
            'content': '''
class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if self.root is None:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
            '''.strip(),
            'score': 0.72,
            'metadata': {'file_path': 'data_structures/bst.py', 'language': 'python'}
        }
    ]
    
    # Test 1: Initialize reranker
    print("1. Initializing subscription reranker...")
    reranker = SubscriptionHaikuReRanker()
    
    stats = reranker.get_stats()
    print(f"   Subscription mode: {stats['subscription_mode']}")
    print(f"   SDK available: {stats['sdk_available']}")
    print()
    
    # Test 2: Simple reranking
    print("2. Testing simple reranking...")
    query = "efficient algorithm implementation"
    
    start_time = time.perf_counter()
    reranked_results = await reranker.rerank_simple(
        query=query,
        results=sample_results,
        context="Looking for high-performance Python algorithms",
        max_results=3
    )
    elapsed_time = time.perf_counter() - start_time
    
    print(f"   Query: {query}")
    print(f"   Processing time: {elapsed_time*1000:.1f}ms")
    print(f"   Results returned: {len(reranked_results)}")
    print()
    
    # Test 3: Show results
    print("3. Reranked results:")
    for i, result in enumerate(reranked_results):
        metadata = result.get('metadata', {})
        confidence = result.get('reranking_confidence', 0.0)
        subscription_mode = metadata.get('subscription_mode', False)
        
        print(f"   {i+1}. File: {metadata.get('file_path', 'unknown')}")
        print(f"      Score: {result.get('score', 0.0):.3f}")
        print(f"      Confidence: {confidence:.3f}")
        print(f"      Used subscription: {subscription_mode}")
        print(f"      Content preview: {result['content'][:100]}...")
        print()
    
    # Test 4: Performance stats
    print("4. Performance statistics:")
    final_stats = reranker.get_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    print()
    
    # Test 5: Convenience function
    print("5. Testing convenience function...")
    convenience_results = await rerank_with_subscription(
        query="data structure implementation",
        results=sample_results[:2],
        max_results=2
    )
    
    print(f"   Convenience function returned {len(convenience_results)} results")
    print(f"   First result: {convenience_results[0].get('metadata', {}).get('file_path', 'unknown')}")
    print()
    
    print("✅ Subscription reranker test completed!")
    
    return reranked_results, final_stats

if __name__ == "__main__":
    # Run the test
    results, stats = asyncio.run(test_subscription_reranker())
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    print(f"Subscription mode active: {stats['subscription_mode']}")
    print(f"Total requests: {stats['requests']}")
    print(f"Subscription API calls: {stats['subscription_calls']}")
    print(f"Mock calls: {stats['mock_calls']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Average processing time: {stats['avg_processing_time']*1000:.1f}ms")