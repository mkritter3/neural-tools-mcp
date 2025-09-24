#!/usr/bin/env python3
"""
Test Tree-sitter symbol extraction
"""

import asyncio
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent / 'neural-tools/src/servers/services'))

from tree_sitter_extractor import TreeSitterExtractor

async def test_extraction():
    extractor = TreeSitterExtractor()

    # Test Python file
    test_file = Path(__file__).parent / "neural-tools/src/servers/services/neo4j_service.py"

    with open(test_file, 'r') as f:
        content = f.read()

    result = await extractor.extract_symbols_from_file(
        str(test_file),
        content,
        timeout=5.0
    )

    if result.get('error'):
        print(f"❌ Error: {result['error']}")
    else:
        symbols = result.get('symbols', [])
        print(f"✅ Extracted {len(symbols)} symbols from {test_file.name}")

        # Count by type
        types = {}
        for symbol in symbols:
            sym_type = symbol.get('type', 'unknown')
            types[sym_type] = types.get(sym_type, 0) + 1

        print("\nSymbol breakdown:")
        for sym_type, count in sorted(types.items()):
            print(f"  {sym_type}: {count}")

        # Show some examples
        print("\nExample symbols:")
        for symbol in symbols[:5]:
            print(f"  - {symbol.get('type')}: {symbol.get('name')} at line {symbol.get('line')}")

    print(f"\nExtractor stats: {extractor.stats}")

if __name__ == "__main__":
    asyncio.run(test_extraction())