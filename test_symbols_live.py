#!/usr/bin/env python3
"""Test file for live symbol extraction verification"""

class SymbolExtractionTest:
    """Test class for verifying tree-sitter extraction"""
    
    def __init__(self):
        self.status = "initialized"
    
    def process(self, data):
        """Process some data"""
        return len(data)
    
    async def async_method(self):
        """An async method"""
        return "async result"

def standalone_function(x, y):
    """Add two numbers"""
    return x + y

async def async_standalone():
    """Async standalone function"""
    import asyncio
    await asyncio.sleep(0.1)
    return "done"

# Added to trigger file change detection
def new_test_function():
    """New function to test symbol extraction"""
    return "testing tree-sitter"