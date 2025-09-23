#!/usr/bin/env python3
"""
Simple test script to verify Graphiti works with local Ollama - no OpenAI API keys needed
"""

import asyncio
import os
import logging
from datetime import datetime
from pathlib import Path

# Set up environment to avoid OpenAI dependency
os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_local_graphiti():
    """Test Graphiti with local Ollama setup"""

    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.llm_client import LLMConfig
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        logger.info("✅ Graphiti imports successful")

        # Configure LLM to use local Ollama
        llm_config = LLMConfig(
            api_key="dummy",  # Ollama doesn't need real API key
            model="graphiti-coder-optimized:latest",
            small_model="graphiti-coder-optimized:latest",
            base_url="http://localhost:11434/v1"  # Local Ollama
        )

        llm_client = OpenAIGenericClient(config=llm_config)
        logger.info("✅ LLM client configured for local Ollama")

        # Configure embedder to use local Ollama
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="dummy",
                embedding_model="nomic-embed-text",
                embedding_dim=768,
                base_url="http://localhost:11434/v1"
            )
        )
        logger.info("✅ Embedder configured for local Ollama")

        # Create Graphiti client with local Neo4j
        client = Graphiti(
            "bolt://localhost:47687",  # Local Neo4j port
            "neo4j",
            "graphrag-password",
            llm_client=llm_client,
            embedder=embedder
        )
        logger.info("✅ Graphiti client created")

        # Test connection and build indices
        await client.build_indices_and_constraints()
        logger.info("✅ Neo4j indices and constraints built")

        # Test adding a simple episode
        result = await client.add_episode(
            name=f"test_episode_{datetime.now().isoformat()}",
            episode_body="This is a test of local Graphiti with Ollama integration",
            source_description="Local test script",
            reference_time=datetime.now(),
            group_id="test-project"
        )
        logger.info(f"✅ Episode added successfully: {result}")

        # Test search
        search_results = await client.search(
            query="test integration",
            group_id="test-project"
        )
        logger.info(f"✅ Search completed, found {len(search_results)} results")

        await client.close()
        logger.info("✅ All tests passed - Graphiti works with local Ollama!")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_local_graphiti())
    exit(0 if success else 1)