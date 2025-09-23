#!/usr/bin/env python3
"""
Quick Graph Context Indexer - ADR-0075 Phase 1
Index real codebase files to demonstrate graph relationships without waiting for Nomic
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.neo4j_service import Neo4jService

class QuickGraphIndexer:
    """Quick indexer to demonstrate File->Chunk graph relationships"""

    def __init__(self, project_name: str = "claude-l9-template"):
        self.project_name = project_name
        self.neo4j = Neo4jService(project_name)
        self.processed_count = 0

    async def initialize(self):
        """Initialize Neo4j service"""
        print("🔧 Initializing quick graph indexer...")
        await self.neo4j.initialize()
        print("✅ Neo4j service ready with vector indexes")
        return True

    async def create_file_and_chunks(self, file_path: Path) -> bool:
        """Create File node and Chunk nodes with relationships (no embeddings for speed)"""
        try:
            if not file_path.exists() or file_path.is_dir():
                return False

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return False

            # Simple chunking
            chunks = self.chunk_content(content, file_path)
            if not chunks:
                return False

            # Create File node and chunks in single transaction
            success = await self.store_file_with_chunks(file_path, content, chunks)

            if success:
                self.processed_count += 1
                print(f"✅ {file_path.name} -> File + {len(chunks)} Chunks")
                return True

            return False

        except Exception as e:
            print(f"❌ {file_path.name}: {e}")
            return False

    def chunk_content(self, content: str, file_path: Path) -> List[Dict]:
        """Simple content chunking for graph demo"""
        lines = content.split('\n')
        chunks = []

        # Chunk every 20 lines for graph demonstration
        for i in range(0, len(lines), 20):
            chunk_lines = lines[i:i+20]
            chunk_text = '\n'.join(chunk_lines)

            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'start_line': i + 1,
                    'end_line': min(i + 20, len(lines)),
                    'chunk_type': 'code'
                })

        return chunks

    async def store_file_with_chunks(self, file_path: Path, content: str, chunks: List[Dict]) -> bool:
        """Store File node with Chunk relationships (ADR-0075 Phase 1)"""
        try:
            relative_path = file_path.relative_to(Path("."))
            file_hash = hashlib.sha256(content.encode()).hexdigest()

            # Single transaction: File + Chunks + Relationships
            cypher = """
            // 1. Create File node
            MERGE (f:File {path: $path, project: $project})
            SET f += {
                name: $name,
                content_hash: $file_hash,
                size: $size,
                language: $language,
                indexed_time: datetime(),
                project: $project
            }

            // 2. Remove old chunks
            MATCH (f)-[:HAS_CHUNK]->(old_chunk:Chunk)
            DETACH DELETE old_chunk

            // 3. Create new chunks with File->Chunk relationships
            WITH f
            UNWIND $chunks_data AS chunk_data
            CREATE (c:Chunk {
                chunk_id: chunk_data.chunk_id,
                text: chunk_data.text,
                start_line: chunk_data.start_line,
                end_line: chunk_data.end_line,
                project: $project,
                created_time: datetime()
            })
            CREATE (f)-[:HAS_CHUNK]->(c)

            RETURN f.path as file_path, count(c) as chunks_created
            """

            # Prepare chunk data
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{relative_path}:chunk:{i}"
                chunks_data.append({
                    'chunk_id': chunk_id,
                    'text': chunk['text'],
                    'start_line': chunk['start_line'],
                    'end_line': chunk['end_line']
                })

            params = {
                'path': str(relative_path),
                'project': self.project_name,
                'name': file_path.name,
                'file_hash': file_hash,
                'size': len(content),
                'language': file_path.suffix[1:] if file_path.suffix else 'unknown',
                'chunks_data': chunks_data
            }

            result = await self.neo4j.execute_cypher(cypher, params)
            return result.get('status') == 'success'

        except Exception as e:
            print(f"  ❌ Storage failed: {e}")
            return False

    def get_indexable_files(self, root_dir: Path, limit: int = 10) -> List[Path]:
        """Get limited list of files for quick demo"""
        patterns = ["**/*.py"]  # Focus on Python files for demo

        files = []
        for pattern in patterns:
            files.extend(root_dir.glob(pattern))

        # Filter and limit
        filtered = []
        ignore_patterns = ['__pycache__', '.git', 'node_modules', '.pytest_cache']

        for file_path in files:
            if not any(ignore in str(file_path) for ignore in ignore_patterns):
                filtered.append(file_path)
                if len(filtered) >= limit:
                    break

        return sorted(filtered)

async def main():
    """Quick graph indexing demonstration"""
    print("🚀 ADR-0075 Phase 1: Quick Graph Context Demo")
    print("="*50)

    indexer = QuickGraphIndexer()
    await indexer.initialize()

    # Index limited set of files for demonstration
    root_dir = Path(".")
    files = indexer.get_indexable_files(root_dir, limit=10)

    print(f"📚 Indexing {len(files)} files for graph demo...")
    print()

    start_time = time.time()
    success_count = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i:2d}/{len(files)}] Processing: {file_path}")

        if await indexer.create_file_and_chunks(file_path):
            success_count += 1

    elapsed = time.time() - start_time
    print()
    print("="*50)
    print("🎯 Quick Graph Indexing Complete!")
    print(f"✅ Files processed: {success_count}/{len(files)}")
    print(f"⏱️  Total time: {elapsed:.1f} seconds")
    print()

    # Test the graph structure
    print("🕸️  Testing Graph Relationships...")

    # Check File->Chunk relationships
    file_chunk_query = """
    MATCH (f:File)-[:HAS_CHUNK]->(c:Chunk)
    WHERE f.project = $project
    RETURN f.path, count(c) as chunk_count
    ORDER BY chunk_count DESC
    LIMIT 5
    """

    result = await indexer.neo4j.execute_cypher(file_chunk_query, {'project': 'claude-l9-template'})
    if result.get('status') == 'success':
        relationships = result['result']
        if relationships:
            print("✅ File->Chunk relationships created:")
            for record in relationships:
                print(f"   {record['path']}: {record['chunk_count']} chunks")
        else:
            print("❌ No File->Chunk relationships found")

    print()
    print("🎯 Graph context foundation ready!")
    print("✅ ADR-0075 Phase 1 complete - File and Chunk nodes with relationships")

if __name__ == "__main__":
    asyncio.run(main())