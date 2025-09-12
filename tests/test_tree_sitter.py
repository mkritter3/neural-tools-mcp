#!/usr/bin/env python3
"""
Test tree-sitter symbol extraction functionality
"""

import asyncio
import sys
from pathlib import Path

# Add services directory to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools/src/servers/services"))

from tree_sitter_extractor import TreeSitterExtractor

# Sample Python code
PYTHON_CODE = '''
class ExampleClass:
    """A sample class for testing"""
    
    def __init__(self, name: str):
        """Initialize with name"""
        self.name = name
    
    def greet(self) -> str:
        """Return greeting message"""
        return f"Hello, {self.name}!"
    
    async def async_method(self, value: int) -> int:
        """An async method"""
        await asyncio.sleep(0.1)
        return value * 2

def standalone_function(x: int, y: int) -> int:
    """Add two numbers"""
    return x + y

async def async_function():
    """An async function"""
    await asyncio.sleep(0.1)
    return "done"
'''

# Sample JavaScript code
JAVASCRIPT_CODE = '''
class Calculator {
    constructor(initialValue = 0) {
        this.value = initialValue;
    }
    
    add(x) {
        this.value += x;
        return this;
    }
    
    multiply(x) {
        this.value *= x;
        return this;
    }
}

function processData(data) {
    return data.map(item => item * 2);
}

const arrowFunction = (a, b) => {
    return a + b;
};

export const exportedFunc = () => {
    console.log("Exported function");
};
'''

# Sample TypeScript code
TYPESCRIPT_CODE = '''
interface User {
    id: number;
    name: string;
    email?: string;
}

type Status = "active" | "inactive" | "pending";

class UserService {
    private users: User[] = [];
    
    constructor(private apiUrl: string) {}
    
    async getUser(id: number): Promise<User | null> {
        const user = this.users.find(u => u.id === id);
        return user || null;
    }
    
    addUser(user: User): void {
        this.users.push(user);
    }
}

function genericFunction<T>(value: T): T {
    return value;
}
'''

async def test_extraction():
    """Test tree-sitter extraction on sample code"""
    
    extractor = TreeSitterExtractor()
    
    print("=" * 60)
    print("🌳 Tree-Sitter Symbol Extraction Test")
    print("=" * 60)
    
    # Test Python extraction
    print("\n📍 Testing Python extraction...")
    py_result = await extractor.extract_symbols_from_file(
        "test.py", 
        PYTHON_CODE,
        timeout=5.0
    )
    
    if py_result['error']:
        print(f"❌ Python extraction failed: {py_result['error']}")
    else:
        symbols = py_result['symbols']
        print(f"✅ Extracted {len(symbols)} Python symbols:")
        for symbol in symbols:
            indent = "  " if symbol['type'] == 'method' else ""
            print(f"{indent}- {symbol['type']}: {symbol['name']} (lines {symbol['start_line']}-{symbol['end_line']})")
            if symbol.get('docstring'):
                print(f"{indent}  Docstring: {symbol['docstring'][:50]}...")
    
    # Test JavaScript extraction
    print("\n📍 Testing JavaScript extraction...")
    js_result = await extractor.extract_symbols_from_file(
        "test.js",
        JAVASCRIPT_CODE,
        timeout=5.0
    )
    
    if js_result['error']:
        print(f"❌ JavaScript extraction failed: {js_result['error']}")
    else:
        symbols = js_result['symbols']
        print(f"✅ Extracted {len(symbols)} JavaScript symbols:")
        for symbol in symbols:
            arrow = " (arrow)" if symbol.get('is_arrow') else ""
            print(f"- {symbol['type']}: {symbol['name']}{arrow} (lines {symbol['start_line']}-{symbol['end_line']})")
    
    # Test TypeScript extraction
    print("\n📍 Testing TypeScript extraction...")
    ts_result = await extractor.extract_symbols_from_file(
        "test.ts",
        TYPESCRIPT_CODE,
        timeout=5.0
    )
    
    if ts_result['error']:
        print(f"❌ TypeScript extraction failed: {ts_result['error']}")
    else:
        symbols = ts_result['symbols']
        print(f"✅ Extracted {len(symbols)} TypeScript symbols:")
        for symbol in symbols:
            print(f"- {symbol['type']}: {symbol['name']} (lines {symbol['start_line']}-{symbol['end_line']})")
    
    # Test batch extraction
    print("\n📍 Testing batch extraction...")
    files = [
        ("sample1.py", PYTHON_CODE),
        ("sample2.js", JAVASCRIPT_CODE),
        ("sample3.ts", TYPESCRIPT_CODE)
    ]
    
    batch_results = await extractor.extract_symbols_batch(files, batch_size=2)
    
    total_symbols = sum(len(r['symbols']) for r in batch_results.values() if not r.get('error'))
    print(f"✅ Batch extraction complete: {total_symbols} total symbols from {len(files)} files")
    
    # Print statistics
    print("\n📊 Extraction Statistics:")
    stats = extractor.get_statistics()
    for key, value in stats.items():
        if key == 'languages':
            print(f"  Languages processed:")
            for lang, lang_stats in value.items():
                print(f"    - {lang}: {lang_stats['files']} files, {lang_stats['symbols']} symbols")
        else:
            print(f"  {key}: {value}")
    
    print("\n✨ Tree-sitter extraction test complete!")

if __name__ == "__main__":
    asyncio.run(test_extraction())