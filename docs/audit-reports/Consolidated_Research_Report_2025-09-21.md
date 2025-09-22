# Consolidated Research Report (2025-09-21)

**Author:** Gemini Advanced Auditing Agent
**Status:** Complete

## 1. Introduction

This document is a complimentary compilation of the web research performed during the architectural analysis and planning for the advanced RAG system. It serves as the evidentiary backbone for the recommendations made in the main report, "Comprehensive Analysis Report (v5, Final Integration Guide)". The research is organized thematically.

---

## 2. Core Architecture & Technologies (ADRs 66-68)

### 2.1. Neo4j 5 Vector Search Performance

**Query:** `Neo4j 5 vector search performance benchmarks 2025`

**Findings:** Neo4j 5's strength lies in its ability to combine vector search with its powerful graph capabilities for rich, context-aware results. While it may not outperform specialized vector databases in raw similarity search speed, its hybrid approach is particularly beneficial for Retrieval-Augmented Generation (RAG) pipelines, enhancing accuracy and explainability. Real-world applications have shown significant improvements in various industries. Performance can be tuned via HNSW parameters.

### 2.2. Late Chunking for Embeddings

**Query:** `Late Chunking for embeddings research paper arXiv:2409.04701v3`

**Findings:** The research paper "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models" (arXiv:2409.04701v3) by researchers from Jina AI and Weaviate introduces this novel method. The core idea is to embed all tokens of a long text first and apply chunking *after* the transformer model, allowing the resulting chunk embeddings to capture the full contextual information from the entire document. This leads to superior results across various retrieval tasks and can be applied to a wide range of long-context embedding models without requiring additional training.

### 2.3. Graphiti Temporal Knowledge Graphs

**Query:** `getzep/graphiti open source project review and status 2025`

**Findings:** As of 2025, `getzep/graphiti` is a rapidly evolving and impactful open-source Python framework for building real-time, temporally-aware knowledge graphs for AI agents. It features a bi-temporal data model, efficient hybrid retrieval (semantic, keyword, graph), and support for custom entity definitions using Pydantic. The project is under active development with frequent releases and has gained significant attention. Benchmarks indicate substantial performance improvements over traditional approaches, with a P95 query latency of ~300ms.

### 2.4. Alternatives to Graphiti

**Query:** `alternatives to Graphiti for temporal knowledge graphs open source`

**Findings:** Graphiti is a strong, specialized open-source solution. Alternatives are generally lower-level or for different use cases. They include using the underlying graph databases directly (like Neo4j, FalkorDB, Kuzu) and modeling the temporal aspects manually, or using other Graph RAG implementations that would require custom development to add the explicit temporality that Graphiti provides out-of-the-box.

### 2.5. Detecting Outdated Documentation

**Query:** `detecting outdated documentation using knowledge graphs research 2025`

**Findings:** Research indicates that knowledge graphs are a key technology for maintaining the freshness of documentation. The focus is on proactive maintenance through automated knowledge graph generation and real-time data processing. By representing documents and code as nodes with explicit relationships, KGs allow for the identification of inconsistencies or gaps. The use of bitemporal stamping and event-driven updates is a core research area that directly supports the continuous updating of information, preventing documentation from becoming outdated.

---

## 3. Advanced RAG Practices (Google Best Practices)

### 3.1. General RAG Best Practices

**Query:** `Google Cloud RAG best practices 2025`

**Findings:** Google Cloud's best practices emphasize robust evaluation, advanced retrieval, and leveraging specialized AI services. Key takeaways include: adopting hybrid retrieval (keyword + vector), implementing domain-aware "smart chunking", using metadata for ranking, and employing query reformulation. For performance, caching and asynchronous pipelines are recommended. Security is paramount, with a zero-trust model and PII redaction. The Vertex AI suite (Agent Builder, Vector Search, RAG Engine) is the primary toolchain.

### 3.2. Agentic and Self-Correcting RAG

**Query:** `Google agentic RAG and self-correcting RAG pipelines`

**Findings:** This is the frontier of RAG. **Agentic RAG** uses an LLM to orchestrate the RAG pipeline, allowing it to handle complex, multi-step workflows and decide which tools to use (e.g., vector search, web search, API calls). **Self-Correcting RAG** (also known as CRAG or Self-RAG) adds a feedback loop. It includes a retrieval evaluator to grade the relevance of retrieved documents. If the quality is low, the system can automatically rewrite the query or trigger alternative search methods to find better information, significantly reducing hallucinations and improving accuracy.

---

## 4. Open-Source Model & Framework Landscape

### 4.1. LLMs for Agentic Control (Function Calling)

**Query:** `open source LLM for function calling and tool use 2025`

**Findings:** As of 2025, many powerful open-source LLMs have native function-calling/tool-use capabilities. Top models include Meta's Llama 3 family, Mistral AI's models (Mistral Large 2, Mixtral-8x7B), Falcon 2, and Cohere's Command R+. Specialized models like NexusRaven-V2 and Functionary are also available. These models can be run locally and are supported by frameworks like LangChain, LlamaIndex, and Ollama, making them ideal for building agentic systems without external API calls.

### 4.2. Cross-Encoder Models for Re-ranking

**Query:** `open source cross-encoder re-ranking models sentence-transformers`

**Findings:** The `sentence-transformers` library provides excellent support for local, open-source cross-encoder models. These models are highly accurate for re-ranking a smaller set of candidate documents retrieved by a faster first-stage retriever. The most commonly recommended and effective model for this task is `cross-encoder/ms-marco-MiniLM-L-6-v2`, which is small, fast, and fine-tuned on a massive dataset for semantic search.

### 4.3. RAG Evaluation Frameworks

**Query:** `open source RAG evaluation framework Ragas TruLens 2025`

**Findings:** Ragas and TruLens are the leading open-source frameworks for evaluating RAG systems. **Ragas** is noted for its suite of metrics (faithfulness, answer relevancy, context recall) and its ability to perform "reference-free" evaluation, often using an LLM as a judge. **TruLens** focuses on evaluating response groundedness and context relevance via "feedback functions". Both are actively developed, integrate with LangChain/LlamaIndex, and can be configured to use local LLMs, avoiding data leakage and external API calls.

### 4.4. Qwen Model Family Deep Dive

**Queries:** `Qwen 30b-3a model`, `Qwen2 model family function calling`, `Qwen2-32B-Instruct performance benchmarks`

**Findings:** The user's query "Qwen 30b-3a" refers to the `Qwen3-30B-A3B` model, a Mixture-of-Experts (MoE) model from Alibaba. It has 30.5B total parameters but only activates ~3.3B during inference, making it extremely efficient. The Qwen family (Qwen2, Qwen3) has excellent, native support for function calling. Benchmarks for models in this class (e.g., Qwen2.5-32B-Instruct) show very strong performance in coding, math, and reasoning, competitive with other top-tier open-source models. The combination of high performance, tool use, and MoE efficiency makes it a prime candidate for local deployment.

---

## 5. Deployment & Operations

### 5.1. Ollama Docker Deployment with GPU

**Queries:** `official ollama docker image gpu support`, `docker compose ollama gpu nvidia container toolkit`

**Findings:** The official `ollama/ollama` Docker image supports GPU acceleration on Linux hosts via the NVIDIA Container Toolkit. The correct and modern way to enable this in Docker Compose is to use a `deploy` section in the service definition to reserve GPU resources. This allows the container to access the host's GPU for significantly faster inference.

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              count: all
```

### 5.2. Inter-Container Communication

**Query:** `connect to ollama service from another docker container docker-compose`

**Findings:** When multiple services are defined in the same `docker-compose.yml` file, Docker Compose automatically creates a shared network. Services can communicate with each other by using the service name as a hostname. For example, a Python application in a service named `app` can connect to an Ollama service named `ollama_service` at the URL `http://ollama_service:11434`. This is the standard and recommended practice for inter-container communication.
