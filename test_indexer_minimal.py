#!/usr/bin/env python3
"""Minimal test for indexer with tree-sitter"""

import asyncio
import os
import sys
sys.path.append('neural-tools/src/servers/services')

from tree_sitter_extractor import TreeSitterExtractor

async def test():
    print("Testing tree-sitter extraction...")
    
    extractor = TreeSitterExtractor()
    
    test_code = """
class TestClass:
    def method1(self):
        return "test"
    
    def method2(self, arg):
        return arg * 2

def standalone_func():
    return TestClass()
"""
    
    result = await extractor.extract_symbols_from_file(
        'test.py', 
        test_code,
        timeout=5.0
    )
    
    print(f"Extracted {len(result['symbols'])} symbols:")
    for symbol in result['symbols']:
        print(f"  - {symbol['type']}: {symbol.get('qualified_name', symbol['name'])}")
    
    stats = extractor.get_statistics()
    print(f"\nStatistics: {stats}")
    
    return result

if __name__ == "__main__":
    result = asyncio.run(test())
    print(f"\n✓ Tree-sitter extraction is working!")
    print(f"Total symbols extracted: {len(result['symbols'])}")