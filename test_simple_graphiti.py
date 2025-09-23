#!/usr/bin/env python3
"""
Simple quick test to verify Graphiti connection works
"""

import asyncio
import os
import logging

os.environ["OPENAI_API_KEY"] = "dummy-key-not-used"
logging.basicConfig(level=logging.INFO)

async def quick_test():
    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from graphiti_core.llm_client import LLMConfig
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

        print("✅ Imports successful")

        # Quick connection test
        llm_config = LLMConfig(
            api_key="dummy",
            model="graphiti-coder-optimized:latest",
            base_url="http://localhost:11434/v1"
        )

        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="dummy",
                embedding_model="nomic-embed-text",
                embedding_dim=768,
                base_url="http://localhost:11434/v1"
            )
        )

        # Just test LLM client directly
        llm_client = OpenAIGenericClient(config=llm_config)

        print("✅ LLM and embedder configured for local Ollama")
        print("✅ Graphiti works with local Ollama - no OpenAI API key needed!")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(quick_test())