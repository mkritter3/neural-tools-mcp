## 1. Executive Summary

This definitive report provides a complete, end-to-end guide for deploying a production-grade, advanced RAG architecture. It addresses the critical challenge of ensuring reliable, structured JSON output from local LLMs by incorporating the state-of-the-art `Outlines` library to guarantee schema adherence.

This guide provides a clear, non-disruptive path to enhance your system by:
1.  **Detailing the Complete System:** It now includes a full architectural diagram showing how the **Indexing Pipeline (Write Path)** and the **Querying Pipeline (Read Path)** work together.
2.  **Ensuring Production-Grade Reliability:** It introduces a multi-layered strategy to guarantee structured JSON output for all machine-to-machine communication within the RAG pipeline.
3.  **Integrating with Existing Infrastructure:** It details how to leverage your pre-existing `redis-cache` service and add the new AI service alongside your current `gemma-tagger`.
4.  **Providing Actionable Code for Integration:** It includes a complete `docker-compose.yml` setup and detailed Python code for using the Qwen3 model, the `Outlines` library for guaranteed JSON, and Redis for robust caching.

By following this guide, the project can evolve into a state-of-the-art, self-correcting RAG system that is not only powerful but also exceptionally reliable.

---

## 2. The Complete System Architecture: Indexing and Querying

The full architecture consists of two distinct, asynchronous pipelines that operate on the same central **Neo4j + Graphiti Knowledge Graph**.

1.  **The Indexing Pipeline (Write Path):** Its sole job is to watch for new or changed files and write that information into the knowledge graph. This is based on your existing indexer architecture (ADRs 30, 66, 67).
2.  **The Querying Pipeline (Read Path):** Its sole job is to take a user query and use the information *already in* the knowledge graph to generate an answer. This is the enhanced agentic RAG system.

### How They Integrate

The key integration point is that the **Indexing Pipeline** will now use the same AI subagent as the Querying Pipeline to enrich the data *at write time*. When the `EnhancedIndexerService` processes a file, it will call the **Qwen3 Relationship Detector** to pre-calculate and store the relationships between code and documentation in the knowledge graph. This makes the graph richer and subsequent queries faster and more accurate.

### Full System Diagram

```
+---------------------------------------------------------------------------------+
|                                  WRITE PATH                                     |
|                           (The Indexing Pipeline)                               |
+---------------------------------------------------------------------------------+
|                                                                                 |
|  [File Change] -> [IndexerOrchestrator] -> Spawns [EnhancedIndexerService]      |
|                                                              |                  |
|                                                              v                  |
|  +---------------------------------------------------------------------------+  |
|  | EnhancedIndexerService for file 'x'                                     |  |
|  |  1. Chunks file content                                                   |  |
|  |  2. Generates embeddings (calls Nomic service)                            |  |
|  |  3. Calls Qwen3 Relationship Detector -> Gets back structured JSON       |  |
|  |  4. Bundles everything into a Graphiti JSON Episode                       |  |
|  +---------------------------------------------------------------------------+  |
|                                        |                                        |
|                                        v                                        |
+===================================[ GRAPHITI ]==================================+
|                                                                                 |
|                     CENTRAL NEO4J KNOWLEDGE GRAPH DATABASE                      |
|                  (Stores Chunks, Embeddings, and Relationships)                 |
|                                                                                 |
+=================================================================================+
|                                        ^                                        |
|                                        |                                        |
|  +---------------------------------------------------------------------------+  |
|  | The Querying Pipeline (from v6 Report)                                  |  |
|  |  1. Qwen3 Orchestrator plans the query                                    |  |
|  |  2. Retrieves context from Neo4j (including pre-calculated relationships)|  |
|  |  3. Refiner re-ranks results                                              |  |
|  |  4. Corrector validates and potentially re-queries with temporal fallback |  |
|  |  5. Synthesizes final natural language answer                             |  |
|  +---------------------------------------------------------------------------+  |
|                                      ^                                          |
|                                      |                                          |
|                                [User Query]                                     |
|                                                                                 |
+---------------------------------------------------------------------------------+
|                                   READ PATH                                     |
|                            (The Querying Pipeline)                              |
+---------------------------------------------------------------------------------+
```

---

## 3. Guiding Principle: When to Use Structured vs. Natural Language Output

A critical design decision is when to enforce structured JSON output versus when to allow for free-form natural language. The guiding principle is as follows:

> **Use structured JSON output whenever the LLM's response is consumed by another machine/software component. Use natural language output only for the final step where the answer is presented directly to the human user.**

Applying this to our architecture:

- **The Orchestrator:** **MUST** produce structured JSON so the application can parse its plan and execute tool calls.
- **The Relationship Detector:** **MUST** produce structured JSON so the output can be reliably used to create a Graphiti knowledge graph episode.
- **The Corrector (Grader):** **MUST** produce structured JSON to provide a clear, parsable signal (e.g., `{"relevance": "POOR"}`) to the control loop.
- **The Final Answer Generator:** **MUST NOT** produce structured JSON. Its sole purpose is to synthesize all the structured data into a helpful, readable paragraph for the user.

---

## 4. Definitive End-to-End Integration Plan

(The Docker Compose, Dockerfile, and initialization steps remain the same as v5.)

---

## 5. Ensuring Reliable Structured JSON Output

Your question about the reliability of structured JSON output is critical. A model failing to produce valid JSON is a major failure point. To build a production-grade system, we will implement a multi-layered strategy that guarantees correct output for all internal components.

### Layer 1: Prompting and Native Ollama Features

As a baseline, we will use Ollama's native `format: "json"` feature and a strong system prompt that instructs the model to only output JSON. This is a good first step, but we will not rely on it alone.

### Layer 2 (Recommended): Guaranteed Schema with `Outlines`

For maximum reliability, we will integrate the `Outlines` library. Unlike other tools that validate JSON *after* the model generates it (and retry on failure), `Outlines` works at a lower level. It constrains the model during the generation process, guiding it to only produce tokens that fit a predefined Pydantic schema. **This guarantees that the output will always be valid JSON that matches our structure.**

### Layer 3: Resilient Error Handling

As a final safety net, the application code will include `try/except` blocks to handle any unexpected issues (like network errors to the Ollama container) and will gracefully fall back to the heuristic method if the AI-based analysis fails.

---

## 6. Application Integration Guide (with Robust JSON)

This updated guide incorporates the `Outlines` library for guaranteed JSON output and Redis for caching.

### 6.1. Update `app/requirements.txt`

Add the `outlines-core` and `redis` libraries.

```txt
# ... (your other requirements)
httpx
redis
outlines-core
pydantic
```

### 6.2. Update the Relationship Detector Code

This new version of the `HaikuEnhancedRelationshipDetector` is significantly more robust.

```python
import os
import json
import redis
import outlines
from pydantic import BaseModel, Field
from typing import List, Literal
from pathlib import Path

# --- Pydantic Schema for Guaranteed Output ---
# This defines the structure we will force the LLM to follow.
class Relationship(BaseModel):
    type: Literal["DOCUMENTS", "DESCRIBES", "EXAMPLES_FOR", "RELATED_TO"] = Field(..., description="The type of relationship.")
    target_files: List[str] = Field(..., description="The code files this documentation refers to.")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0.")
    reasoning: str = Field(..., description="A brief explanation for the relationship.")

class AnalysisResult(BaseModel):
    relationships: List[Relationship]
    requires_manual_review: bool = Field(..., description="True if the analysis is ambiguous.")

# --- Connection Details from Environment Variables ---
QWEN_API_HOST = os.getenv("QWEN_HOST", "http://localhost:11435")
REDIS_HOST = os.getenv("REDIS_CACHE_HOST", "localhost")
# ... (other redis vars)

class HaikuEnhancedRelationshipDetector:
    """Qwen3 subagent using Outlines for guaranteed JSON and Redis for caching."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.heuristic_detector = BasicHeuristicDetector()
        # Initialize Redis connection (same as v5)
        # ...

        # Initialize the Outlines model, pointing to our local Ollama service
        self.qwen_model = outlines.models.ollama(
            model="qwen2:32b-instruct-q4_K_M",
            host=QWEN_API_HOST.replace("http://", "").split(':')[0],
            port=int(QWEN_API_HOST.split(':')[2])
        )

    async def _analyze_with_qwen3(self, file_path: str, context: dict) -> AnalysisResult:
        """Performs analysis using the local Qwen3 model, with JSON output guaranteed by Outlines."""
        system_prompt = "You are a code relationship analyzer. Your task is to determine relationships between documentation and code files based on the provided context. You must follow the user-provided JSON schema."
        user_prompt = f"""Analyze relationships for the documentation file: {file_path}

CONTEXT:
{context}

Based on the context, determine the accurate doc-code relationships."""

        try:
            # This is the core change. We pass the Pydantic model to Outlines.
            # Outlines will guide the Qwen3 model to generate a valid AnalysisResult object.
            generator = outlines.generate.json(self.qwen_model, AnalysisResult)
            # Note: The exact prompt format may need tuning for the model (e.g., using special tokens)
            structured_result = await generator(f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n")
            return structured_result
        except Exception as e:
            print(f"Error during Outlines generation: {e}")
            return None # Return None on failure

    async def detect_relationships(self, file_path: str, content_sample: str = None) -> list[dict]:
        """Main method with caching and fallback logic."""
        # ... (heuristic check and cache check logic is the same as v5) ...

        print(f"CACHE MISS for {file_path}. Querying Qwen3 model via Outlines.")
        analysis_context = await self._prepare_analysis_context(file_path, content_sample)
        structured_result = await self._analyze_with_qwen3(file_path, analysis_context)

        if not structured_result:
            # Fallback if Outlines/Qwen3 fails for some reason
            return self._format_heuristic_results(heuristic_relationships)

        # Convert Pydantic model to list of dicts for consistent output
        qwen3_relationships = [rel.dict() for rel in structured_result.relationships]

        # Store result in Redis cache
        if self.redis_cache and qwen3_relationships:
            self.redis_cache.set(cache_key, json.dumps(qwen3_relationships), ex=86400)

        return qwen3_relationships

    # ... (other helper methods remain the same)
```
