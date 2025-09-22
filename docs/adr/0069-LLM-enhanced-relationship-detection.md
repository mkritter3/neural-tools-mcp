# ADR-0069: Local LLM Enhanced Relationship Detection for Unified GraphRAG Platform

**Date:** September 21, 2025
**Status:** Proposed
**Tags:** qwen3-30b-a3b, ollama, local-llm, relationship-detection, mcp-enhancement, graphrag-optimization, data-sovereignty
**Builds on:** ADR-0068 (MCP Tools Modernization), ADR-0067 (Graphiti Temporal Knowledge Graphs)
**Dependencies:** Ollama, Qwen3-30B-A3B Model, Outlines Library, Redis Cache

## Executive Summary

Enhance ADR-68's unified content search with local Qwen3-30B-A3B LLM subagents for intelligent document-code relationship detection, reducing false positives in temporal discrepancy analysis from ~15% (heuristic baseline) to <5% (intelligent detection) while ensuring data sovereignty and eliminating external API costs through local Ollama deployment with production-grade reliability via Outlines structured output.

## Context

### Current State (Post-ADR-68)

ADR-68 implements unified content search with basic heuristic relationship detection:

```python
# Current heuristic approach
def detect_relationships_heuristic(file_path: str) -> List[Relationship]:
    relationships = []

    if file_path.endswith('.md'):
        # File naming heuristics: auth.md → auth.py
        base_name = Path(file_path).stem
        potential_code_files = find_files_matching(f"{base_name}.*")

        # Directory structure: docs/feature/ → src/feature/
        feature_dir = Path(file_path).parent.name
        related_src = find_files_in_directory(f"src/{feature_dir}/")

        relationships.extend(create_relationships(potential_code_files + related_src))

    return relationships
```

### Limitations of Heuristic Detection

1. **High False Positive Rate**: ~15% incorrect relationship detection
2. **Missed Complex Relationships**: Tutorial docs → multiple implementation files
3. **Refactoring Brittleness**: Fails when file names diverge from documentation
4. **Limited Semantic Understanding**: Cannot detect conceptual relationships
5. **External API Dependency**: Previous solutions require external service calls, increasing costs and latency

### Why Qwen3-30B-A3B Local LLM for Enhancement

Based on comprehensive analysis of local LLM deployment options, Qwen3-30B-A3B via Ollama provides optimal characteristics for this use case:

**From Research Analysis:**
- **Data Sovereignty**: Complete local processing eliminates external API dependencies
- **Performance**: 78 tokens/sec local inference vs 52-68 tokens/sec via external APIs
- **MoE Efficiency**: 30.5B total parameters with only 3.3B active per inference
- **Function Calling**: Native support for structured JSON output via Qwen-Agent framework
- **Apache 2.0 License**: Permissive licensing for commercial deployment
- **Isolated Context**: "Isolated context windows" prevent cross-contamination
- **SDK Integration**: Native Python SDK with streaming and session management

## Decision

**Implement local Qwen3-Coder-30B-A3B LLM subagents via Ollama to enhance relationship detection accuracy while ensuring data sovereignty and eliminating external API costs through local deployment with production-grade reliability guarantees.**

## Technical Architecture

### Enhanced Relationship Detection Pipeline

```python
# neural-tools/src/servers/tools/llm_enhanced_relationship_detector.py
"""
Qwen3-30B-A3B via Ollama integration for intelligent document-code relationship detection
"""

import os
import json
import redis
import httpx
import outlines
from pathlib import Path
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

# Pydantic schemas for guaranteed structured output via Outlines
class Relationship(BaseModel):
    type: Literal["DOCUMENTS", "DESCRIBES", "EXAMPLES_FOR", "RELATED_TO"] = Field(..., description="The type of relationship.")
    target_files: List[str] = Field(..., description="The code files this documentation refers to.")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0.")
    reasoning: str = Field(..., description="A brief explanation for the relationship.")

class AnalysisResult(BaseModel):
    relationships: List[Relationship]
    requires_manual_review: bool = Field(..., description="True if the analysis is ambiguous.")

# Environment configuration
QWEN_API_HOST = os.getenv("QWEN_HOST", "http://qwen_orchestrator:11434")
REDIS_HOST = os.getenv("REDIS_CACHE_HOST", "redis-cache")
REDIS_PORT = int(os.getenv("REDIS_CACHE_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_CACHE_PASSWORD")

class LLMEnhancedRelationshipDetector:
    """Qwen3-30B-A3B local LLM subagent for intelligent doc-code relationships with guaranteed JSON output"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.heuristic_detector = BasicHeuristicDetector()

        # Redis cache integration for persistent, shared caching
        self._initialize_redis_cache()

        # Initialize Outlines model for guaranteed structured output
        self._initialize_qwen3_model()

    def _initialize_redis_cache(self):
        """Initialize Redis cache with graceful fallback"""
        try:
            self.redis_cache = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            self.redis_cache.ping()  # Verify connection
            print(f"✅ Connected to Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            print(f"⚠️ Redis cache unavailable: {e}. Using in-memory fallback.")
            self.redis_cache = None

    def _initialize_qwen3_model(self):
        """Initialize Qwen3 model via Outlines for guaranteed JSON output"""
        try:
            # Extract host and port from QWEN_API_HOST
            host_parts = QWEN_API_HOST.replace("http://", "").split(":")
            host = host_parts[0]
            port = int(host_parts[1]) if len(host_parts) > 1 else 11434

            self.qwen_model = outlines.models.ollama(
                model="qwen3-coder-30b-a3b",
                host=host,
                port=port
            )

            # Create structured generator for guaranteed JSON output
            self.json_generator = outlines.generate.json(self.qwen_model, AnalysisResult)
            print(f"✅ Qwen3-Coder-30B-A3B model initialized via Outlines at {QWEN_API_HOST}")
        except Exception as e:
            print(f"❌ Failed to initialize Qwen3 model: {e}")
            self.qwen_model = None
            self.json_generator = None

    def _get_relationship_detection_prompt(self) -> str:
        """Optimized system prompt for deterministic relationship detection"""
        return """You are a code relationship analyzer. Your task is to determine relationships between documentation and code files.

TASK: Analyze file content and structure to identify accurate doc-code relationships.

OUTPUT FORMAT (JSON):
{
  "relationships": [
    {
      "type": "DOCUMENTS|DESCRIBES|EXPLAINS|TUTORIAL_FOR|EXAMPLES_FOR",
      "target_files": ["path/to/code.py", "path/to/other.js"],
      "confidence": 0.95,
      "reasoning": "Brief explanation"
    }
  ],
  "requires_manual_review": false
}

RULES:
1. Only suggest relationships with >80% confidence
2. Use file content, not just names, for analysis
3. Distinguish between DOCUMENTS (comprehensive) vs DESCRIBES (partial)
4. Set requires_manual_review=true for ambiguous cases
5. Be conservative - false negatives better than false positives"""

    async def detect_relationships(self, file_path: str, content_sample: str = None) -> List[Dict]:
        """
        Enhanced relationship detection with local Qwen3-30B-A3B intelligence

        Strategy:
        1. Fast heuristic detection first (80% of cases)
        2. Qwen3 analysis for ambiguous cases (20% of cases)
        3. Redis caching for persistent results
        4. Guaranteed JSON output via Outlines
        """
        try:
            # Step 1: Fast heuristic detection
            heuristic_relationships = await self.heuristic_detector.detect(file_path)

            # Step 2: Determine if LLM analysis needed
            if not self._needs_intelligent_analysis(file_path, heuristic_relationships):
                return self._format_heuristic_results(heuristic_relationships)

            # Step 3: Check Redis cache for similar patterns
            cache_key = self._get_cache_key(file_path, content_sample)
            if self.redis_cache:
                cached_result = self.redis_cache.get(cache_key)
                if cached_result:
                    print(f"✅ Cache hit for {file_path}")
                    return json.loads(cached_result)

            # Step 4: Qwen3 analysis for complex cases with guaranteed JSON output
            print(f"🧠 LLM analysis required for {file_path}")
            qwen3_results = await self._analyze_with_qwen3(file_path, content_sample)

            # Step 5: Cache results with expiration (24 hours)
            if self.redis_cache and qwen3_results:
                self.redis_cache.set(cache_key, json.dumps(qwen3_results), ex=86400)

            return qwen3_results

        except Exception as e:
            logger.warning(f"LLM relationship detection failed for {file_path}: {e}")
            # Graceful fallback to heuristic detection
            return self._format_heuristic_results(heuristic_relationships)

    def _needs_intelligent_analysis(self, file_path: str, heuristic_results: List) -> bool:
        """Determine if Haiku analysis is warranted"""
        # Triggers for Haiku analysis:
        return any([
            len(heuristic_results) == 0,  # No heuristic matches found
            len(heuristic_results) > 5,   # Too many potential matches (ambiguous)
            "tutorial" in file_path.lower(),  # Complex tutorial relationships
            "example" in file_path.lower(),   # Example code relationships
            file_path.endswith(('.rst', '.adoc')),  # Complex doc formats
            self._has_conflicting_heuristics(heuristic_results)
        ])

    async def _analyze_with_qwen3(self, file_path: str, content_sample: str = None) -> List[Dict]:
        """Perform Qwen3-30B-A3B analysis with guaranteed JSON output via Outlines"""

        if not self.json_generator:
            print("❌ Qwen3 model not available, falling back to heuristics")
            return []

        # Prepare context for LLM analysis
        analysis_context = await self._prepare_analysis_context(file_path, content_sample)

        # Construct system and user prompts optimized for Qwen3
        system_prompt = """You are a code relationship analyzer. Your task is to determine relationships between documentation and code files based on the provided context. You must follow the user-provided JSON schema."""

        user_prompt = f"""Analyze relationships for the documentation file: {file_path}

FILE CONTENT SAMPLE:
{analysis_context['content_preview']}

POTENTIAL CODE FILES:
{analysis_context['nearby_code_files']}

HEURISTIC SUGGESTIONS:
{analysis_context['heuristic_suggestions']}

Based on the context, determine the accurate doc-code relationships. Be conservative - false negatives are better than false positives."""

        try:
            # Use Outlines to generate guaranteed structured output
            # Format prompt for Qwen3's chat template
            full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

            # Generate structured response - guaranteed to match AnalysisResult schema
            structured_result = self.json_generator(full_prompt)

            # Convert Pydantic model to list of dicts for consistent output
            if structured_result and structured_result.relationships:
                return [rel.dict() for rel in structured_result.relationships]
            else:
                return []

        except Exception as e:
            print(f"❌ Qwen3 analysis failed: {e}")
            return []

    async def _prepare_analysis_context(self, file_path: str, content_sample: str = None) -> Dict:
        """Prepare optimized context for LLM analysis"""

        # Read file content if not provided (first 2KB for efficiency)
        if content_sample is None:
            try:
                with open(self.project_path / file_path, 'r', encoding='utf-8') as f:
                    content_sample = f.read(2048)  # First 2KB
            except Exception:
                content_sample = ""

        # Find nearby code files efficiently
        file_dir = Path(file_path).parent
        nearby_code_files = []

        for pattern in ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c', '*.go']:
            nearby_code_files.extend(file_dir.glob(pattern))

        # Limit to most relevant files
        nearby_code_files = [str(f) for f in nearby_code_files[:10]]

        return {
            'content_preview': content_sample[:1000],  # First 1KB
            'nearby_code_files': nearby_code_files,
            'heuristic_suggestions': await self.heuristic_detector.detect(file_path)
        }

    def _parse_haiku_response(self, response_text: str) -> List[Dict]:
        """Parse Haiku JSON response with error handling"""
        try:
            import json
            import re

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())

                # Validate and format results
                relationships = parsed.get('relationships', [])
                return self._validate_haiku_results(relationships)

        except Exception as e:
            logger.warning(f"Failed to parse Haiku response: {e}")

        # Fallback to empty results
        return []

    def _validate_haiku_results(self, relationships: List[Dict]) -> List[Dict]:
        """Validate Haiku results and apply confidence thresholds"""
        validated = []

        for rel in relationships:
            # Only accept high-confidence relationships
            if rel.get('confidence', 0) >= 0.8:
                # Verify target files exist
                valid_targets = []
                for target in rel.get('target_files', []):
                    if (self.project_path / target).exists():
                        valid_targets.append(target)

                if valid_targets:
                    rel['target_files'] = valid_targets
                    validated.append(rel)

        return validated

    def _get_cache_key(self, file_path: str, content_sample: str = None) -> str:
        """Generate cache key for similar file patterns"""
        import hashlib

        # Create cache key from file pattern and content hash
        file_pattern = f"{Path(file_path).suffix}_{Path(file_path).parent.name}"
        content_hash = hashlib.md5((content_sample or "")[:500].encode()).hexdigest()[:8]

        return f"{file_pattern}_{content_hash}"

# Integration with existing unified content search
class EnhancedUnifiedContentSearch:
    """Enhanced version of ADR-68 unified search with Haiku integration"""

    def __init__(self, service_container):
        self.container = service_container
        self.graphiti = service_container.graphiti
        self.haiku_detector = HaikuEnhancedRelationshipDetector(
            service_container.project_path
        )

    async def unified_content_search(
        self,
        query: str,
        limit: int = 10,
        content_filter: str = "auto",
        detect_discrepancies: bool = True,
        use_haiku_enhancement: bool = True
    ) -> Dict:
        """
        Enhanced unified search with optional Haiku relationship detection
        """
        # Execute base unified search (from ADR-68)
        base_results = await self.graphiti.search(
            query=query,
            num_results=limit,
            hybrid_search=True,
            content_filter=content_filter
        )

        # Enhanced temporal discrepancy detection
        if detect_discrepancies:
            if use_haiku_enhancement:
                # Use Haiku for high-accuracy relationship detection
                enhanced_relationships = await self._detect_relationships_haiku(base_results)
                discrepancies = await self._analyze_temporal_discrepancies_enhanced(
                    base_results, enhanced_relationships
                )
            else:
                # Fallback to heuristic detection (ADR-68 baseline)
                heuristic_relationships = await self._detect_relationships_heuristic(base_results)
                discrepancies = await self._analyze_temporal_discrepancies_basic(
                    base_results, heuristic_relationships
                )
        else:
            discrepancies = []

        return {
            "results": base_results,
            "discrepancies": discrepancies,
            "metadata": {
                "query": query,
                "search_method": "unified_graphrag_haiku_enhanced",
                "relationship_detection": "haiku" if use_haiku_enhancement else "heuristic",
                "haiku_usage_count": getattr(self.haiku_detector, 'usage_count', 0)
            }
        }

    async def _detect_relationships_haiku(self, search_results: List[Dict]) -> List[Dict]:
        """Detect relationships using Haiku enhancement"""
        relationships = []

        for result in search_results:
            if result.get('content_type') in ['markdown', 'documentation']:
                file_relationships = await self.haiku_detector.detect_relationships(
                    result['file_path'],
                    result.get('content', '')[:1000]  # First 1KB
                )
                relationships.extend(file_relationships)

        return relationships
```

## Performance Optimization Strategy

### Selective Haiku Usage

```python
# Cost optimization through intelligent triggering
class HaikuUsageOptimizer:
    """Optimize Haiku usage for cost and performance"""

    def __init__(self):
        self.daily_usage_limit = 1000  # Haiku calls per day
        self.current_usage = 0
        self.circuit_breaker_threshold = 0.02  # 2% of file processing

    async def should_use_haiku(self, file_path: str, context: Dict) -> bool:
        """Determine if Haiku analysis is cost-justified"""

        # Circuit breaker: Limit Haiku usage
        if self.current_usage >= self.daily_usage_limit:
            return False

        # Priority cases that justify Haiku cost
        high_priority = any([
            context.get('user_query_priority') == 'high',
            'critical' in file_path.lower(),
            context.get('discrepancy_count', 0) > 5,  # Many potential issues
            context.get('manual_review_requested', False)
        ])

        # Standard cases: Use heuristics to decide
        if not high_priority:
            # Skip if heuristics are confident
            heuristic_confidence = context.get('heuristic_confidence', 0)
            if heuristic_confidence > 0.9:
                return False

        return True

# Batch processing for efficiency
async def batch_relationship_detection(files: List[str]) -> Dict[str, List[Dict]]:
    """Process multiple files in optimized batches"""

    # Group similar files for batch processing
    batches = group_files_by_similarity(files)
    results = {}

    for batch in batches:
        # Single Haiku session for related files
        async with ClaudeSDKClient(options=haiku_options) as client:
            for file_path in batch:
                relationships = await analyze_file_relationships(client, file_path)
                results[file_path] = relationships

    return results
```

## Implementation Plan

### Phase 1: Docker Infrastructure Setup (Week 1)

**Goals**: Deploy Qwen3-Coder-30B-A3B via Ollama with production-grade reliability

```yaml
# docker-compose.yml additions - Hybrid deployment configuration
services:
  # Optional Ollama service for shared/Docker deployment
  # Use profiles to enable only when needed
  qwen_orchestrator:
    image: ollama/ollama:latest
    container_name: qwen_orchestrator
    profiles: ["docker-ollama"]  # Only runs with --profile docker-ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - qwen_models:/root/.ollama
    ports:
      - "11435:11434"  # Dedicated port for Qwen3 orchestrator
    restart: unless-stopped
    networks:
      - default
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Enhanced l9-graphrag service with flexible Qwen3 configuration
  l9-graphrag:
    environment:
      # Existing environment variables...
      # Flexible Qwen host - supports both local and Docker deployment
      - QWEN_HOST=${QWEN_HOST:-http://qwen_orchestrator:11434}
      - REDIS_CACHE_HOST=redis-cache
      - REDIS_CACHE_PORT=6379
      - REDIS_CACHE_PASSWORD=${REDIS_CACHE_PASSWORD:-cache-secret-key}
    depends_on:
      - redis-cache
      # Conditional dependency only when using Docker Ollama
      # qwen_orchestrator: (managed by profiles)

volumes:
  qwen_models:
    driver: local
```

```python
# Week 1 Deliverables
deliverables = [
    "LLMEnhancedRelationshipDetector class with Ollama integration",
    "Outlines library integration for guaranteed JSON output",
    "Redis caching system for persistent results",
    "Hybrid Docker Compose configuration with deployment profiles",
    "Model initialization and health monitoring",
    "Local development setup documentation"
]

# Week 1 Testing
unit_testing_criteria = [
    "Qwen3 model successfully deployed and accessible (both modes)",
    "GPU utilization <80% during inference (host mode)",
    "Redis cache connectivity and persistence",
    "Outlines JSON schema validation 100% success rate",
    "Docker container startup time <60 seconds (Docker mode)",
    "Host connectivity via host.docker.internal (local mode)"
]

# Week 1 Integration Tests
integration_testing_criteria = [
    "End-to-end relationship detection pipeline test",
    "Real documentation file processing with actual Qwen3 inference",
    "Redis cache persistence across container restarts",
    "Graceful fallback from LLM to heuristic on Qwen3 failure",
    "Memory usage <4GB during sustained operation",
    "Cross-platform deployment validation (local + Docker modes)"
]

# Week 1 CI/CD Gates
cicd_exit_conditions = [
    "✅ Automated Docker health checks pass",
    "✅ Integration test suite achieves 100% pass rate",
    "✅ Performance benchmarks meet latency targets (<100ms)",
    "✅ Resource utilization stays within limits",
    "✅ Zero memory leaks during 1-hour stress test",
    "✅ Documentation deployment guides work for new users"
]
```

## Deployment Configuration

### Local Development Setup (Host Ollama + GPU)

**Recommended for fast local development with GPU acceleration**

```bash
# 1. Install and run Ollama on host machine
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve --host 0.0.0.0

# 2. Load your local Qwen3-Coder model (one time setup)
# Create Modelfile for your local GGUF model
cat > Modelfile << EOF
FROM /Users/mkr/models/qwen3-coder/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
EOF

# Create the model in Ollama
ollama create qwen3-coder-30b-a3b -f Modelfile

# 3. Create local environment configuration
cat > .env.local << EOF
QWEN_HOST=http://host.docker.internal:11434
REDIS_CACHE_HOST=redis-cache
REDIS_CACHE_PORT=6379
REDIS_CACHE_PASSWORD=cache-secret-key
EOF

# 4. Start services (excludes Docker Ollama)
docker-compose --env-file .env.local up
```

### Shared/Team Deployment (Docker Ollama)

**For team members or servers with proper GPU Docker support**

```bash
# 1. Create Docker environment configuration
cat > .env.docker << EOF
QWEN_HOST=http://qwen_orchestrator:11434
REDIS_CACHE_HOST=redis-cache
REDIS_CACHE_PORT=6379
REDIS_CACHE_PASSWORD=cache-secret-key
EOF

# 2. Start all services including Docker Ollama
docker-compose --profile docker-ollama --env-file .env.docker up

# 3. Initialize model (first time only)
# Option A: Pull from Ollama Hub (if available)
docker exec qwen_orchestrator ollama pull qwen3-coder:30b

# Option B: Copy your local model into container
docker cp /Users/mkr/models/qwen3-coder/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf qwen_orchestrator:/tmp/
docker exec qwen_orchestrator sh -c "
cat > /tmp/Modelfile << EOF
FROM /tmp/qwen3-coder-30b-a3b-instruct-q4_k_m.gguf
TEMPLATE \"\"\"<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
\"\"\"
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
EOF
ollama create qwen3-coder-30b-a3b -f /tmp/Modelfile
"
```

### CPU-Only Fallback (Development/Testing)

**For environments without GPU access**

```bash
# Use smaller, CPU-friendly model for testing
QWEN_MODEL=qwen3-coder:7b  # Much faster on CPU
QWEN_HOST=http://host.docker.internal:11434
```

### Phase 2: Integration with ADR-68 (Week 2)

**Goals**: Seamless integration with existing unified content search

```python
# Integration Points
integration_steps = [
    "Enhance unified_content_search with local LLM analysis",
    "A/B testing framework for accuracy comparison",
    "GPU resource monitoring and thermal management",
    "Configuration flags for gradual rollout",
    "Comprehensive error handling and Redis fallback"
]

# A/B Testing Configuration
ab_testing_config = {
    "llm_percentage": 20,  # Start with 20% LLM usage
    "baseline_percentage": 80,  # 80% heuristic baseline
    "success_metrics": [
        "relationship_detection_accuracy",
        "false_positive_rate",
        "inference_latency",
        "gpu_utilization_efficiency",
        "redis_cache_hit_rate"
    ]
}

# Week 2 Integration Tests
phase2_integration_tests = [
    "ADR-68 unified_content_search integration verification",
    "A/B testing framework statistical significance validation",
    "Cross-ADR compatibility: ADR-66 (Neo4j) + ADR-67 (Graphiti) + ADR-68 + ADR-69",
    "Real-world codebase processing (>1000 files) end-to-end test",
    "Concurrent user simulation (10+ simultaneous queries)",
    "Database consistency validation after LLM enrichment"
]

# Week 2 CI/CD Gates
phase2_cicd_gates = [
    "✅ A/B testing shows statistically significant accuracy improvement",
    "✅ Performance regression tests pass (no degradation to existing features)",
    "✅ Multi-ADR integration tests achieve 100% success rate",
    "✅ Load testing demonstrates stable performance under concurrent usage",
    "✅ Backward compatibility maintained with existing ADR-68 API",
    "✅ Monitoring dashboards accurately track LLM vs heuristic performance"
]
```

### Phase 3: Production Deployment (Week 3)

**Goals**: Full production deployment with monitoring

```python
# Production Readiness Checklist
production_checklist = [
    "✅ Performance validation: <100ms inference latency",
    "✅ Resource efficiency: GPU utilization 60-80% range",
    "✅ Accuracy improvement: >95% relationship detection",
    "✅ Data sovereignty: Zero external API calls",
    "✅ Fallback reliability: 100% graceful degradation",
    "✅ Monitoring and alerting: Comprehensive coverage",
    "✅ Documentation: User guides and troubleshooting",
    "✅ Circuit breaker: Automatic cost protection"
]

# Week 3 Production Integration Tests
production_integration_tests = [
    "Full-scale deployment validation in production environment",
    "Disaster recovery testing (Redis failure, Qwen3 unavailability)",
    "24-hour continuous operation stability test",
    "Production data processing validation (real user workloads)",
    "Security testing: data isolation, no credential exposure",
    "Multi-project concurrent operation validation",
    "Rollback procedure validation and timing verification"
]

# Week 3 Production CI/CD Gates
production_cicd_gates = [
    "✅ Production deployment automation succeeds without manual intervention",
    "✅ Health checks and monitoring confirm system stability over 24+ hours",
    "✅ User acceptance testing demonstrates measurable accuracy improvements",
    "✅ Security audit confirms no data sovereignty violations",
    "✅ Performance benchmarks sustained under production load",
    "✅ Incident response procedures tested and documented",
    "✅ Rollback capability verified with <5 minute recovery time"
]

# Gradual Rollout Strategy
rollout_strategy = {
    "week_1": "Internal testing (5% traffic)",
    "week_2": "Beta users (25% traffic)",
    "week_3": "Full rollout (100% traffic with fallback)"
}
```

## Success Metrics and Validation

### Performance Targets

| Metric | Baseline (Heuristic) | Target (Haiku Enhanced) | Measurement Method |
|--------|---------------------|-------------------------|-------------------|
| **Relationship Detection Accuracy** | ~85% | >95% | Manual validation dataset |
| **False Positive Rate** | ~15% | <5% | Automated validation |
| **Processing Latency** | <50ms | <100ms | Real-time monitoring |
| **Cost per Relationship** | $0.001 | <$0.005 | SDK cost tracking |
| **Haiku Usage Rate** | 0% | <2% of files | Usage analytics |

### Quality Metrics

```python
# Automated quality validation
class RelationshipQualityValidator:
    """Validate relationship detection quality"""

    async def validate_accuracy(self, test_dataset: List[Dict]) -> Dict:
        """Measure accuracy against ground truth"""
        results = {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

        for test_case in test_dataset:
            predicted = await self.haiku_detector.detect_relationships(
                test_case['file_path'],
                test_case['content']
            )

            ground_truth = test_case['expected_relationships']

            # Calculate metrics
            tp, fp, fn = self._compare_relationships(predicted, ground_truth)
            results["true_positives"] += tp
            results["false_positives"] += fp
            results["false_negatives"] += fn

        # Calculate precision, recall, F1
        precision = results["true_positives"] / (results["true_positives"] + results["false_positives"])
        recall = results["true_positives"] / (results["true_positives"] + results["false_negatives"])
        f1_score = 2 * (precision * recall) / (precision + recall)

        results.update({
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })

        return results
```

## Risk Mitigation and Fallback Strategy

### Circuit Breaker Implementation

```python
class HaikuCircuitBreaker:
    """Protect against excessive Haiku usage and costs"""

    def __init__(self):
        self.daily_cost_limit = 10.0  # $10/day Haiku budget
        self.current_cost = 0.0
        self.error_threshold = 0.1  # 10% error rate threshold
        self.consecutive_errors = 0
        self.is_open = False

    async def check_availability(self) -> bool:
        """Check if Haiku usage should be allowed"""

        # Cost-based circuit breaker
        if self.current_cost >= self.daily_cost_limit:
            logger.warning("Haiku daily cost limit reached, falling back to heuristics")
            self.is_open = True
            return False

        # Error-based circuit breaker
        if self.consecutive_errors >= 5:
            logger.warning("Too many Haiku errors, falling back to heuristics")
            self.is_open = True
            return False

        return not self.is_open

    async def record_usage(self, cost: float, success: bool):
        """Record Haiku usage for circuit breaker logic"""
        self.current_cost += cost

        if success:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
```

### Graceful Degradation

```python
async def relationship_detection_with_fallback(
    file_path: str,
    content: str = None
) -> List[Dict]:
    """Relationship detection with automatic fallback"""

    try:
        # Check circuit breaker
        if await circuit_breaker.check_availability():
            # Attempt Haiku analysis
            relationships = await haiku_detector.detect_relationships(file_path, content)
            await circuit_breaker.record_usage(cost=0.002, success=True)
            return relationships

    except Exception as e:
        logger.warning(f"Haiku analysis failed for {file_path}: {e}")
        await circuit_breaker.record_usage(cost=0.002, success=False)

    # Automatic fallback to heuristic detection
    logger.info(f"Falling back to heuristic detection for {file_path}")
    return await heuristic_detector.detect_relationships(file_path)
```

## Configuration and Environment Setup

### SDK Configuration Optimization

```python
# Optimized Claude Code SDK setup for production
def get_production_haiku_config() -> ClaudeCodeOptions:
    """Production-optimized Haiku configuration"""

    return ClaudeCodeOptions(
        # Model selection
        model="claude-3-5-haiku-20241022",

        # Performance optimization
        max_turns=1,  # Single-turn for deterministic tasks
        max_thinking_tokens=2000,  # Reduced for faster response

        # Tool restrictions for security
        allowed_tools=["Read", "Glob", "Grep"],
        disallowed_tools=["Bash", "Write", "WebSearch"],

        # Permission handling
        permission_mode="acceptEdits",  # No interactive prompts

        # Session optimization
        cwd=None,  # Set per-request
        add_dirs=[],  # Minimal context

        # Error handling
        extra_args={
            "--timeout": "30",  # 30s timeout for Haiku calls
            "--max-retries": "2"  # Limited retries
        }
    )

# Environment-specific configuration
haiku_config = {
    "development": {
        "daily_usage_limit": 10000,
        "cost_limit": 50.0,
        "enable_debug_logging": True
    },
    "staging": {
        "daily_usage_limit": 5000,
        "cost_limit": 25.0,
        "enable_debug_logging": False
    },
    "production": {
        "daily_usage_limit": 1000,
        "cost_limit": 10.0,
        "enable_debug_logging": False,
        "enable_circuit_breaker": True
    }
}
```

## Integration with Existing ADR Infrastructure

### Compatibility with ADR-66/67/68

```python
# Seamless integration with existing architecture
class GraphRAGWithHaikuEnhancement:
    """Integration layer for Haiku enhancement with existing GraphRAG stack"""

    def __init__(self, service_container):
        self.container = service_container
        self.neo4j = service_container.neo4j  # ADR-66
        self.graphiti = service_container.graphiti  # ADR-67
        self.unified_search = service_container.unified_search  # ADR-68
        self.haiku_detector = HaikuEnhancedRelationshipDetector(  # ADR-69
            service_container.project_path
        )

    async def enhanced_unified_search(self, **kwargs) -> Dict:
        """Drop-in replacement for ADR-68 unified search with Haiku enhancement"""

        # Check if enhancement is enabled
        use_haiku = kwargs.pop('use_haiku_enhancement', True)

        if use_haiku and await self._should_use_haiku_enhancement():
            # Enhanced search with Haiku
            return await self._haiku_enhanced_search(**kwargs)
        else:
            # Fallback to ADR-68 baseline
            return await self.unified_search.unified_content_search(**kwargs)

    async def _should_use_haiku_enhancement(self) -> bool:
        """Determine if Haiku enhancement should be used"""
        return all([
            await self.haiku_detector.circuit_breaker.check_availability(),
            self.container.config.get('enable_haiku_enhancement', True),
            not self.container.is_fallback_mode
        ])
```

## Monitoring and Observability

### Performance Monitoring

```python
# Comprehensive monitoring for Haiku usage
class HaikuPerformanceMonitor:
    """Monitor Haiku performance and usage patterns"""

    def __init__(self):
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_cost": 0.0,
            "avg_latency_ms": 0.0,
            "accuracy_rate": 0.0,
            "fallback_rate": 0.0
        }

    async def record_call(self, success: bool, latency_ms: float, cost: float):
        """Record Haiku call metrics"""
        self.metrics["total_calls"] += 1
        self.metrics["total_cost"] += cost

        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1

        # Update running averages
        self.metrics["avg_latency_ms"] = (
            (self.metrics["avg_latency_ms"] * (self.metrics["total_calls"] - 1) + latency_ms)
            / self.metrics["total_calls"]
        )

        self.metrics["fallback_rate"] = (
            self.metrics["failed_calls"] / self.metrics["total_calls"]
        )

    def get_dashboard_metrics(self) -> Dict:
        """Get metrics for monitoring dashboard"""
        return {
            "haiku_usage": {
                "calls_today": self.metrics["total_calls"],
                "success_rate": self.metrics["successful_calls"] / max(1, self.metrics["total_calls"]),
                "cost_today": self.metrics["total_cost"],
                "avg_latency": self.metrics["avg_latency_ms"],
                "fallback_rate": self.metrics["fallback_rate"]
            }
        }
```

## Documentation and Training

### User Guide Integration

```markdown
# Enhanced Relationship Detection (ADR-69)

## Quick Start

The unified content search now includes optional Haiku enhancement:

```python
# Basic usage (automatic enhancement)
results = await unified_content_search(
    "authentication implementation",
    detect_discrepancies=True
)

# Explicit control
results = await unified_content_search(
    "authentication implementation",
    detect_discrepancies=True,
    use_haiku_enhancement=True  # or False for heuristic baseline
)
```

## Configuration

Enable/disable Haiku enhancement in project settings:

```json
{
  "neural_tools": {
    "haiku_enhancement": {
      "enabled": true,
      "daily_usage_limit": 1000,
      "cost_limit": 10.0,
      "accuracy_threshold": 0.8
    }
  }
}
```

## Troubleshooting

Common issues and solutions:

1. **High Haiku Usage**: Check circuit breaker settings
2. **Accuracy Issues**: Review relationship validation thresholds
3. **Cost Concerns**: Adjust usage limits and selective triggering
```

## Decision Outcome

**Status**: PROPOSED

This enhancement provides significant accuracy improvements for document-code relationship detection while maintaining cost-effectiveness through intelligent usage patterns and SDK optimization.

### Core Benefits

1. **Accuracy Improvement**: Relationship detection accuracy from ~85% to >95%
2. **Cost Control**: <2% of total processing requires Haiku analysis
3. **Performance**: <50ms additional latency for enhanced detection
4. **Reliability**: Automatic fallback ensures system robustness
5. **SDK Integration**: Leverages optimized Claude Code SDK patterns

### Integration Summary

| Component | Integration Method | Benefit |
|-----------|-------------------|---------|
| **ADR-66** | Independent - storage layer unchanged | No impact on vector consolidation |
| **ADR-67** | Enhanced - better relationships for Graphiti | Improved temporal relationship accuracy |
| **ADR-68** | Extended - drop-in enhancement option | Backward compatible with fallback |

**Timeline**: 3 weeks implementation + 1 week validation
**Risk**: Low - Optional enhancement with automatic fallback
**ROI**: High - Significant accuracy improvement with controlled costs

---

**Conclusion**: ADR-69 enhances the GraphRAG platform with intelligent relationship detection while preserving the cost-effectiveness and reliability established by previous ADRs. The Claude Code SDK integration provides production-ready patterns for safe and efficient Haiku usage.