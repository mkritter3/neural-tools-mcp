# Comprehensive Analysis Report (v8, Final Guide for macOS Deployment)

**Date:** September 21, 2025
**Author:** Gemini Advanced Auditing Agent
**Status:** Definitive Version

## 1. Executive Summary

This definitive report provides a complete, end-to-end guide for deploying the advanced RAG architecture, specifically tailored for a **macOS environment with an Intel/AMD GPU**. It revises all previous deployment strategies to reflect this critical hardware context.

This guide provides a clear, non-disruptive path to enhance your system by:
1.  **Detailing a Hybrid Deployment Strategy:** To achieve maximum performance, this guide details the correct approach for your hardware: run **Ollama natively on macOS** to leverage GPU acceleration via Metal, and run the application stack (Python, Redis, Neo4j) within Docker.
2.  **Providing a macOS-Specific Guide:** It includes a simplified `docker-compose.yml` and explains how the application container will connect back to the host machine using the `host.docker.internal` DNS name.
3.  **Ensuring Production-Grade Reliability:** The guide retains the multi-layered strategy for guaranteeing structured JSON output using the `Outlines` library, ensuring the application logic is robust.

By following this guide, the project can be deployed efficiently and performantly on your specific hardware.

---

## 2. The Complete System Architecture: Indexing and Querying

(The high-level architecture showing the Write Path and Read Path remains the same as v7.)

---

## 3. Guiding Principle: When to Use Structured vs. Natural Language Output

(The guiding principle remains the same as v7.)

---

## 4. Definitive End-to-End Deployment Guide (macOS / AMD GPU)

This section provides a complete, step-by-step guide for your specific hardware.

### 4.1. Deployment Architecture: Hybrid Approach

Because Docker for Mac does not provide GPU acceleration for AMD GPUs, we will not run Ollama in a container. Instead:
- **Ollama App:** Runs natively on your macOS host to get full access to the Radeon Pro Vega GPU.
- **Application Stack:** The `l9-graphrag` application, Redis, and Neo4j will run in Docker containers managed by Docker Compose.
- **Networking:** The application container will communicate with the native Ollama service via the special Docker DNS name `host.docker.internal`.

### 4.2. Step-by-Step Initialization

**Step 1: Install and Run Ollama on macOS**
1.  Download the official Ollama application for macOS from `ollama.com`.
2.  Install and run the application. You should see the Ollama icon in your macOS menu bar.

**Step 2: Pull the Qwen3 Model (Natively)**
Open your regular macOS terminal (not in Docker) and run the following command. The model will be stored by the native Ollama application.

```bash
ollama pull qwen2:32b-instruct-q4_K_M
```

**Step 3: Verify Native Ollama Service**
In your macOS terminal, verify the model is running.

```bash
curl http://localhost:11434/api/tags
```
You should see a JSON response listing the `qwen2:32b-instruct-q4_K_M` model.

**Step 4: Configure and Run the Docker Environment**

**1. `docker-compose.yml`**
This file is now much simpler. It no longer includes any Ollama services.

```yaml
version: "3.8"

services:
  l9-graphrag:
    build:
      context: ./app
    container_name: rag_app
    # This service connects to other services and to the host machine
    depends_on:
      neo4j:
        condition: service_healthy
      redis-cache:
        condition: service_healthy
    environment:
      # CRITICAL: This tells the app to connect to Ollama on the HOST machine
      - QWEN_HOST=http://host.docker.internal:11434
      
      # Your existing environment variables
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_CACHE_HOST=redis-cache
      - REDIS_CACHE_PORT=6379
      - REDIS_CACHE_PASSWORD=${REDIS_CACHE_PASSWORD:-cache-secret-key}
      # ... other env vars
    networks:
      - default
    restart: unless-stopped

  # Your existing neo4j and redis-cache services remain here
  neo4j:
    # ... (existing configuration)
    networks:
      - default

  redis-cache:
    # ... (existing configuration)
    networks:
      - default

# ... (other services)

volumes:
  # ... (your existing volumes)

networks:
  default:
    name: l9-graphrag-network
```

**2. `app/Dockerfile` and `app/requirements.txt`**
These files remain the same as in the previous guide (v6).

**3. Start the Docker Services**
From your project root, run:

```bash
docker-compose up -d --build
```
This will start your `l9-graphrag` application container, which will now be able to communicate with the Ollama application running natively on your Mac.

---

## 5. Application Integration Guide

The Python code for the application does not need to change, as it correctly reads the `QWEN_HOST` from the environment variable we just configured. When running inside the container, `os.getenv("QWEN_HOST")` will now resolve to `http://host.docker.internal:11434`, correctly pointing to the native Ollama service.

```python
import os

# When this code runs inside the 'l9-graphrag' container, the environment
# variable will be set to point to the host machine.
QWEN_API_HOST = os.getenv("QWEN_HOST", "http://localhost:11434")

# The rest of the Python code for using Outlines, Redis, etc., remains the same.
# ...
```

This hybrid deployment strategy is the most performant and correct solution for your specific hardware and operating system.
