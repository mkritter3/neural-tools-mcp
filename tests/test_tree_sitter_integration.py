#!/usr/bin/env python3
"""Test file for tree-sitter symbol extraction"""

class TreeSitterTest:
    """A test class for symbol extraction"""
    
    def __init__(self, name: str):
        """Initialize with a name"""
        self.name = name
    
    def process_data(self, data: list) -> dict:
        """Process data and return results"""
        result = {}
        for item in data:
            result[item] = len(item)
        return result
    
    async def async_operation(self):
        """An async method for testing"""
        import asyncio
        await asyncio.sleep(0.1)
        return "completed"

def standalone_function(x: int, y: int) -> int:
    """Add two numbers together"""
    return x + y

async def async_main():
    """Main async entry point"""
    test = TreeSitterTest("test")
    result = await test.async_operation()
    print(f"Result: {result}")