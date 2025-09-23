#!/usr/bin/env python3
"""
Test Enhanced Tree-sitter Extraction - ADR-0075 Phase 2
Test the enhanced relationship extraction capabilities
"""

import asyncio
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.tree_sitter_extractor import TreeSitterExtractor

async def test_enhanced_extraction():
    """Test enhanced tree-sitter extraction with relationships"""
    print("🧪 Testing Enhanced Tree-sitter Extraction")
    print("="*50)

    extractor = TreeSitterExtractor()

    # Test Python code with relationships
    test_code = """
import asyncio
from pathlib import Path
from typing import List, Dict

class ServiceContainer:
    def __init__(self, config: Dict):
        self.config = config
        self.neo4j = None

    async def initialize(self):
        \"\"\"Initialize the service container\"\"\"
        await self.setup_neo4j()
        await self.verify_connections()

    async def setup_neo4j(self):
        self.neo4j = Neo4jService(self.config)

    def get_service(self, name: str):
        return self.services.get(name)

class Neo4jService(BaseService):
    def __init__(self, config):
        super().__init__(config)
        self.driver = None

    async def connect(self):
        self.driver = GraphDatabase.driver(self.config['uri'])
"""

    # Extract symbols and relationships
    result = await extractor.extract_symbols_from_file(
        "test_service_container.py",
        test_code
    )

    print(f"📊 Extraction Results:")
    print(f"   Status: {'✅ Success' if not result.get('error') else '❌ ' + result['error']}")
    print(f"   Symbols: {len(result['symbols'])}")
    print()

    # Display symbols
    print("🔍 Extracted Symbols:")
    for i, symbol in enumerate(result['symbols'], 1):
        print(f"   {i}. {symbol['type']} '{symbol['name']}' (line {symbol['start_line']})")
        if symbol.get('parent_class'):
            print(f"      └─ Parent: {symbol['parent_class']}")

    print()
    print("✅ Enhanced tree-sitter extraction test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_extraction())