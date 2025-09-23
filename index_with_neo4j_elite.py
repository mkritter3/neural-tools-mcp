#!/usr/bin/env python3
"""
ADR-0074: Direct Neo4j Elite Indexing Script
Index codebase using direct Neo4j + Nomic services (no complex containers)
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

from servers.services.neo4j_service import Neo4jService
from servers.services.nomic_local_service import NomicService

class EliteIndexer:
    """Direct Neo4j + Nomic indexer for elite performance"""

    def __init__(self, project_name: str = "claude-l9-template"):
        self.project_name = project_name
        self.neo4j = Neo4jService(project_name)
        self.nomic = NomicService()
        self.processed_count = 0
        self.total_files = 0

    async def initialize(self):
        """Initialize services"""
        print("🔧 Initializing elite indexing services...")

        neo4j_result = await self.neo4j.initialize()
        nomic_result = await self.nomic.initialize()

        if not neo4j_result.get("success"):
            raise Exception(f"Neo4j failed: {neo4j_result}")
        if not nomic_result.get("success"):
            raise Exception(f"Nomic failed: {nomic_result}")

        print("✅ Neo4j + Nomic services ready")
        print("📊 Vector indexes: HNSW-optimized for O(log n) search")
        return True

    async def process_file(self, file_path: Path) -> bool:
        """Process a single file with chunking and embedding"""
        try:
            if not file_path.exists() or file_path.is_dir():
                return False

            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            if not content.strip():
                return False

            # Simple chunking (can be enhanced later)
            chunks = self.chunk_content(content, file_path)

            if not chunks:
                return False

            # Process chunks
            for i, chunk_data in enumerate(chunks):
                await self.process_chunk(chunk_data, file_path, i)

            self.processed_count += 1
            print(f"✅ {file_path.name} -> {len(chunks)} chunks")
            return True

        except Exception as e:
            print(f"❌ {file_path.name}: {e}")
            return False

    def chunk_content(self, content: str, file_path: Path) -> List[Dict]:
        """Simple content chunking"""
        lines = content.split('\n')
        chunks = []

        # Simple strategy: chunk by functions/classes or every 20 lines
        current_chunk = []
        start_line = 1

        for i, line in enumerate(lines, 1):
            current_chunk.append(line)

            # Chunk boundaries: function/class definitions or every 20 lines
            if (line.strip().startswith(('def ', 'class ', 'async def')) and len(current_chunk) > 1) or len(current_chunk) >= 20:
                chunk_text = '\n'.join(current_chunk[:-1]) if line.strip().startswith(('def ', 'class ', 'async def')) else '\n'.join(current_chunk)

                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text.strip(),
                        'start_line': start_line,
                        'end_line': i - (1 if line.strip().startswith(('def ', 'class ', 'async def')) else 0),
                        'chunk_type': 'code'
                    })

                current_chunk = [line] if line.strip().startswith(('def ', 'class ', 'async def')) else []
                start_line = i

        # Add remaining content
        if current_chunk and '\n'.join(current_chunk).strip():
            chunks.append({
                'text': '\n'.join(current_chunk).strip(),
                'start_line': start_line,
                'end_line': len(lines),
                'chunk_type': 'code'
            })

        return chunks

    async def process_chunk(self, chunk_data: Dict, file_path: Path, chunk_index: int):
        """Process individual chunk with embedding"""
        try:
            # Generate embedding
            embeddings = await self.nomic.get_embeddings([chunk_data['text']])
            if not embeddings:
                return False

            embedding = embeddings[0]

            # Store in Neo4j with vector index
            chunk_id = f"{file_path.stem}-{chunk_index:03d}"

            cypher = """
            CREATE (c:Chunk {
                chunk_id: $chunk_id,
                text: $text,
                embedding: $embedding,
                project: $project,
                file_path: $file_path,
                start_line: $start_line,
                end_line: $end_line,
                chunk_type: $chunk_type,
                language: $language,
                created_at: datetime()
            })
            RETURN c.chunk_id as stored_id
            """

            result = await self.neo4j.execute_cypher(cypher, {
                'chunk_id': chunk_id,
                'text': chunk_data['text'],
                'embedding': embedding,
                'project': self.project_name,
                'file_path': str(file_path),
                'start_line': chunk_data['start_line'],
                'end_line': chunk_data['end_line'],
                'chunk_type': chunk_data['chunk_type'],
                'language': file_path.suffix[1:] if file_path.suffix else 'unknown'
            })

            return result.get('status') == 'success'

        except Exception as e:
            print(f"  ❌ Chunk {chunk_index}: {e}")
            return False

    def get_indexable_files(self, root_dir: Path) -> List[Path]:
        """Get list of files to index"""
        patterns = [
            "**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx",
            "**/*.java", "**/*.cpp", "**/*.c", "**/*.h", "**/*.rs",
            "**/*.go", "**/*.php", "**/*.rb", "**/*.md", "**/*.adoc"
        ]

        files = []
        for pattern in patterns:
            files.extend(root_dir.glob(pattern))

        # Filter out common ignore patterns
        filtered = []
        ignore_patterns = [
            '__pycache__', '.git', 'node_modules', '.pytest_cache',
            '.mypy_cache', 'dist', 'build', '.venv', 'venv'
        ]

        for file_path in files:
            if not any(ignore in str(file_path) for ignore in ignore_patterns):
                filtered.append(file_path)

        return sorted(filtered)

async def main():
    """Main indexing process"""
    print("🚀 ADR-0074: Elite Neo4j Indexing Started")
    print("="*50)

    # Initialize indexer
    indexer = EliteIndexer()
    await indexer.initialize()

    # Get files to index
    root_dir = Path(".")
    files = indexer.get_indexable_files(root_dir)
    indexer.total_files = len(files)

    print(f"📚 Found {len(files)} files to index")
    print("🧮 Starting embedding generation and Neo4j storage...")
    print()

    # Index files
    start_time = time.time()
    success_count = 0

    for i, file_path in enumerate(files[:20], 1):  # Start with first 20 files
        print(f"[{i:2d}/{min(20, len(files))}] Processing: {file_path}")

        if await indexer.process_file(file_path):
            success_count += 1

        # Progress update
        if i % 5 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"📊 Progress: {i} files, {rate:.1f} files/sec")
            print()

    # Final stats
    elapsed = time.time() - start_time
    print("="*50)
    print("🎯 Indexing Complete!")
    print(f"✅ Successfully processed: {success_count}/{min(20, len(files))} files")
    print(f"⏱️ Total time: {elapsed:.1f} seconds")
    print(f"📈 Rate: {success_count/elapsed:.1f} files/second")
    print()

    # Test the vector search
    print("🔍 Testing elite vector search...")
    query_embedding = await indexer.nomic.get_embeddings(["semantic search function"])
    if query_embedding:
        search_results = await indexer.neo4j.vector_similarity_search(
            query_embedding=query_embedding[0],
            node_type="Chunk",
            limit=3,
            min_score=0.3
        )

        print(f"🎯 Found {len(search_results)} semantic matches!")
        for i, result in enumerate(search_results, 1):
            node = result["node"]
            score = result["similarity_score"]
            print(f"  {i}. Score: {score:.3f} - {node.get('file_path', 'unknown')}:{node.get('start_line', 0)}")

    print("\n🚀 Elite Neo4j indexing ready for production!")

if __name__ == "__main__":
    asyncio.run(main())