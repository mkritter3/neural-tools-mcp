#!/usr/bin/env python3
"""
Find the exact line causing the elite_search error
"""

import asyncio
import sys
import traceback
sys.path.insert(0, '/Users/mkr/local-coding/claude-l9-template/neural-tools/src')

async def test_elite_error():
    """Find the exact error in elite_search"""

    from neural_mcp.tools.elite_search import execute

    args = {
        "query": "how do vector indexes work",
        "limit": 2,
        "max_depth": 1,
        "vector_weight": 0.7,
        "include_explanation": False
    }

    print("Testing elite_search.execute() with args:")
    print(args)
    print("-" * 60)

    try:
        result = await execute(args)
        print("Success! Result:")
        import json
        print(json.dumps(json.loads(result[0].text), indent=2))
    except Exception as e:
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_elite_error())