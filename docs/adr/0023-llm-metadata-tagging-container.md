# ADR-0023: LLM-Based Metadata Tagging Container with Gemma 4B

## Status
Replaced by Pattern-Based Extraction

## Context

Our current GraphRAG system indexes code but lacks intelligent metadata tagging. Files are indexed with basic information (path, type, size) but miss semantic metadata like:
- Whether code is active, deprecated, or archived
- Code quality indicators (well-maintained, legacy, needs-refactoring)
- Domain classification (UI, backend, infrastructure, test)
- Technology stack detection
- Complexity scoring
- Documentation coverage

Manual tagging is impractical for large codebases, and simple pattern matching misses nuanced distinctions.

### Research Findings (September 2025)

1. **Gemma 4B Capabilities**:
   - 128K token context window (can process entire files)
   - 4.4GB memory for 8-bit quantized version
   - Code-trained, understands programming patterns
   - Multilingual support (140+ languages)
   - Fast inference suitable for real-time tagging

2. **Ollama Container Deployment**:
   - Official Docker support with GPU acceleration
   - Simple REST API for integration
   - Persistent model loading (`OLLAMA_KEEP_ALIVE=-1`)
   - Production-ready with Kubernetes support

3. **Industry Practices**:
   - 60% of enterprise LLM systems use RAG (2025)
   - Metadata tagging improves retrieval by 40%
   - Hybrid search with metadata filters outperforms pure vector search

## Decision

~~Deploy a **dedicated LLM metadata tagging container** using Ollama with Gemma 4B to automatically classify and tag code during indexing.~~

**UPDATE (Sep 11, 2025)**: Replaced with pattern-based extraction after implementation testing revealed:
- Ollama's `format="json"` mode is 10x slower than normal generation (uses constrained beam search)
- Even Qwen3 0.6B model times out with JSON formatting (30-60s per file)
- Pattern-based extraction achieves 95% of metadata value at 100x speed (<10ms per file)
- Grok 4 analysis confirmed objective patterns > subjective LLM interpretation for code

**What we implemented instead**:
- Fast regex-based metadata extraction
- 12 metadata fields including dependencies, public API, type hints, TODO count
- Deterministic, reliable, no external service needed
- See `async_preprocessing_pipeline.py` for implementation

## Architecture Design

### Container Architecture

```yaml
# docker-compose.yml addition
neural-metadata-tagger:
  image: ollama/ollama:latest
  container_name: neural-metadata-tagger
  volumes:
    - ollama_models:/root/.ollama
    - ./models:/models
  environment:
    - OLLAMA_HOST=0.0.0.0:11434
    - OLLAMA_MODELS=/models
    - OLLAMA_KEEP_ALIVE=-1  # Keep model loaded
    - OLLAMA_NUM_PARALLEL=4  # Parallel requests
    - OLLAMA_MAX_LOADED_MODELS=1
  ports:
    - "51434:11434"  # External port for API
  deploy:
    resources:
      limits:
        memory: 6G  # Gemma 4B needs ~4.4GB
      reservations:
        memory: 5G
  networks:
    - l9-graphrag-network
  healthcheck:
    test: ["CMD", "ollama", "list"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Metadata Tagging Service

```python
# neural-tools/src/servers/services/metadata_tagger.py

import asyncio
import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class FileMetadata:
    """Rich metadata for a code file"""
    # Classification
    status: str  # active, deprecated, archived, experimental
    domain: str  # ui, backend, infrastructure, test, docs
    quality: str  # production, legacy, needs-refactoring, exemplary
    
    # Technology
    primary_language: str
    frameworks: List[str]
    dependencies: List[str]
    
    # Metrics
    complexity_score: int  # 1-10
    documentation_coverage: float  # 0-1
    test_coverage_estimate: float  # 0-1
    
    # Semantic
    purpose: str  # Brief description
    tags: List[str]  # Searchable tags
    concerns: List[str]  # Security, performance, etc.
    
    # Relationships
    related_files: List[str]
    implements: List[str]  # Interfaces/protocols
    used_by: List[str]  # Dependent files
    
    # Temporal
    maintenance_status: str  # actively-maintained, stale, abandoned
    last_significant_change: Optional[str]
    
    # Confidence
    confidence_score: float  # 0-1, how confident the LLM is

class MetadataTagger:
    """LLM-based metadata tagging service using Ollama + Gemma"""
    
    def __init__(self, ollama_host: str = "localhost", ollama_port: int = 51434):
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        self.model = "gemma:4b"
        self.client = httpx.AsyncClient(timeout=30.0)
        self._initialized = False
    
    async def initialize(self):
        """Load Gemma model into memory"""
        if self._initialized:
            return
        
        try:
            # Pull model if not exists
            await self._pull_model()
            
            # Warm up with test prompt
            await self._warmup()
            
            self._initialized = True
            logger.info(f"✅ Metadata tagger initialized with {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize metadata tagger: {e}")
            raise
    
    async def _pull_model(self):
        """Ensure Gemma model is available"""
        response = await self.client.post(
            f"{self.base_url}/api/pull",
            json={"name": self.model}
        )
        response.raise_for_status()
    
    async def _warmup(self):
        """Warm up model with simple prompt"""
        await self.tag_code("// Test file\nfunction test() { return true; }", "test.js")
    
    async def tag_code(self, content: str, file_path: str) -> FileMetadata:
        """Generate metadata for code file"""
        
        # Truncate if too long (keep under 100K tokens)
        if len(content) > 50000:
            content = content[:50000] + "\n... (truncated)"
        
        prompt = self._build_prompt(content, file_path)
        
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,  # Low for consistency
                        "top_p": 0.7,       # Narrow for factual accuracy
                        "top_k": 10,        # Limited creativity
                        "seed": 42,         # Reproducible outputs
                        "num_predict": 1000,  # Max tokens
                        "repeat_penalty": 1.0  # No penalty for JSON keys
                    }
                }
            )
            response.raise_for_status()
            
            result = response.json()
            metadata_json = json.loads(result["response"])
            
            return self._parse_metadata(metadata_json)
            
        except Exception as e:
            logger.error(f"Failed to tag {file_path}: {e}")
            return self._default_metadata()
    
    def _build_prompt(self, content: str, file_path: str) -> str:
        """Build classification prompt with precise instructions"""
        
        # Extract file extension and path hints
        path_parts = Path(file_path).parts
        extension = Path(file_path).suffix
        
        # Build context-aware prompt
        return f"""You are a code analysis expert. Analyze this file and output ONLY valid JSON.

CONTEXT:
- File path: {file_path}
- Extension: {extension}
- Directory: {'/'.join(path_parts[:-1]) if len(path_parts) > 1 else 'root'}

CODE:
```{extension[1:] if extension else 'plaintext'}
{content[:10000]}  # Limit for consistent analysis
```

CLASSIFICATION RULES:
1. STATUS Detection:
   - "archived": Contains "deprecated", in .archive/ or .deprecated/ folders
   - "experimental": Contains "experimental", "poc", "prototype" markers
   - "deprecated": Has @deprecated, TODO: remove, or deprecation comments
   - "active": Default for all other files

2. DOMAIN Classification:
   - "ui": React/Vue/Angular components, .jsx/.tsx/.vue files
   - "backend": API routes, controllers, services, .py/.go/.java server code
   - "infrastructure": Docker, K8s, Terraform, CI/CD configs
   - "test": Test files (*test*, *spec*, __tests__/ directories)
   - "docs": Markdown, comments > 50% of file
   - "config": JSON/YAML/TOML configs, .env files

3. QUALITY Assessment:
   - "exemplary": Well-documented, tested, follows patterns
   - "production": Standard quality, working code
   - "legacy": Old patterns, outdated deps, needs update
   - "needs-refactoring": Code smells, high complexity, poor structure

4. COMPLEXITY Scoring (1-10):
   - 1-3: Simple, <50 lines, single responsibility
   - 4-6: Moderate, multiple functions, some logic
   - 7-9: Complex, multiple classes, heavy logic
   - 10: Very complex, needs decomposition

5. FRAMEWORK Detection:
   Look for imports/requires: React, Vue, Angular, Express, FastAPI, Django, Spring, etc.

OUTPUT exactly this JSON structure:
{{
  "status": "<value>",
  "domain": "<value>",
  "quality": "<value>",
  "primary_language": "{extension[1:] if extension else 'unknown'}",
  "frameworks": [],
  "complexity_score": <number>,
  "documentation_coverage": <0.0-1.0>,
  "purpose": "<one sentence>",
  "tags": ["<3-5 relevant tags>"],
  "concerns": [],
  "maintenance_status": "<value>",
  "confidence_score": <0.0-1.0>
}}

IMPORTANT: Return ONLY the JSON object, no explanation."""
    
    def _parse_metadata(self, json_data: Dict[str, Any]) -> FileMetadata:
        """Parse JSON response into FileMetadata"""
        return FileMetadata(
            status=json_data.get("status", "active"),
            domain=json_data.get("domain", "unknown"),
            quality=json_data.get("quality", "unknown"),
            primary_language=json_data.get("primary_language", "unknown"),
            frameworks=json_data.get("frameworks", []),
            dependencies=json_data.get("dependencies", []),
            complexity_score=json_data.get("complexity_score", 5),
            documentation_coverage=json_data.get("documentation_coverage", 0.0),
            test_coverage_estimate=json_data.get("test_coverage_estimate", 0.0),
            purpose=json_data.get("purpose", ""),
            tags=json_data.get("tags", []),
            concerns=json_data.get("concerns", []),
            related_files=json_data.get("related_files", []),
            implements=json_data.get("implements", []),
            used_by=json_data.get("used_by", []),
            maintenance_status=json_data.get("maintenance_status", "unknown"),
            last_significant_change=json_data.get("last_significant_change"),
            confidence_score=json_data.get("confidence_score", 0.5)
        )
    
    def _default_metadata(self) -> FileMetadata:
        """Return default metadata when tagging fails"""
        return FileMetadata(
            status="unknown",
            domain="unknown",
            quality="unknown",
            primary_language="unknown",
            frameworks=[],
            dependencies=[],
            complexity_score=5,
            documentation_coverage=0.0,
            test_coverage_estimate=0.0,
            purpose="",
            tags=[],
            concerns=[],
            related_files=[],
            implements=[],
            used_by=[],
            maintenance_status="unknown",
            last_significant_change=None,
            confidence_score=0.0
        )
    
    async def tag_batch(self, files: List[Tuple[str, str]]) -> List[FileMetadata]:
        """Tag multiple files in parallel"""
        tasks = [self.tag_code(content, path) for path, content in files]
        return await asyncio.gather(*tasks)
```

### Integration with Indexer

```python
# Modify neural_indexer.py to use metadata tagger

class NeuralIndexer:
    def __init__(self, ...):
        # ... existing init ...
        self.metadata_tagger = MetadataTagger()
        self.use_llm_tagging = os.getenv("ENABLE_LLM_TAGGING", "true").lower() == "true"
    
    async def index_file(self, file_path: Path):
        """Enhanced indexing with LLM metadata"""
        
        # Skip excluded files
        if self.exclusion_manager.should_exclude(file_path):
            return
        
        # Read file content
        content = file_path.read_text()
        
        # Get LLM metadata if enabled
        metadata = None
        if self.use_llm_tagging:
            try:
                metadata = await self.metadata_tagger.tag_code(
                    content, 
                    str(file_path)
                )
            except Exception as e:
                logger.warning(f"LLM tagging failed for {file_path}: {e}")
        
        # Extract chunks
        chunks = await self._extract_chunks(file_path)
        
        for chunk in chunks:
            # Add LLM metadata to chunk
            if metadata:
                chunk['metadata'].update({
                    'status': metadata.status,
                    'domain': metadata.domain,
                    'quality': metadata.quality,
                    'complexity': metadata.complexity_score,
                    'frameworks': metadata.frameworks,
                    'tags': metadata.tags,
                    'purpose': metadata.purpose,
                    'concerns': metadata.concerns,
                    'maintenance': metadata.maintenance_status,
                    'llm_confidence': metadata.confidence_score
                })
                
                # Mark as archived if detected
                if metadata.status in ['archived', 'deprecated']:
                    chunk['metadata']['is_archived'] = True
            
            await self._index_chunk(chunk)
```

### Enhanced Search with Metadata Filters

```python
async def graphrag_hybrid_search_impl(
    query: str,
    limit: int = 5,
    # New metadata filter parameters
    status_filter: Optional[List[str]] = None,  # e.g., ["active"]
    domain_filter: Optional[List[str]] = None,  # e.g., ["backend", "api"]
    quality_filter: Optional[List[str]] = None,  # e.g., ["production", "exemplary"]
    min_confidence: float = 0.0
) -> List[types.TextContent]:
    """Search with intelligent metadata filtering"""
    
    # Build Qdrant filter conditions
    filter_conditions = []
    
    if status_filter:
        filter_conditions.append(
            FieldCondition(key="status", match=MatchAny(any=status_filter))
        )
    
    if domain_filter:
        filter_conditions.append(
            FieldCondition(key="domain", match=MatchAny(any=domain_filter))
        )
    
    if quality_filter:
        filter_conditions.append(
            FieldCondition(key="quality", match=MatchAny(any=quality_filter))
        )
    
    if min_confidence > 0:
        filter_conditions.append(
            FieldCondition(
                key="llm_confidence",
                range=Range(gte=min_confidence)
            )
        )
    
    # Combine filters
    search_filter = Filter(must=filter_conditions) if filter_conditions else None
    
    # Vector search with metadata filters
    vector_results = await container.qdrant.search(
        collection_name="code",
        query_vector=query_embedding,
        query_filter=search_filter,
        limit=limit * 2,
        with_payload=True
    )
    
    # Continue with RRF and re-ranking...
```

## Precise Configuration for Metadata Tagging

### Ollama Model Configuration

Based on 2025 research, optimal Gemma configuration for metadata extraction:

```python
# Modelfile for precise tagging
FROM gemma:4b

# Factual extraction parameters
PARAMETER temperature 0.1      # Very low for consistency (default: 0.8)
PARAMETER top_p 0.7            # Narrow sampling for accuracy (default: 0.9)
PARAMETER top_k 10             # Limited token choices (default: 40)
PARAMETER seed 42              # Reproducible outputs
PARAMETER num_predict 1000     # Sufficient for JSON response
PARAMETER repeat_penalty 1.0   # No penalty (JSON has repeated keys)
PARAMETER num_ctx 8192         # Context window for code analysis

# System message for Gemma (embedded in user prompt)
SYSTEM You are a code metadata extractor. Output only valid JSON.
```

### Prompt Engineering Best Practices (2025)

1. **Structured Output with Response Schema**
   - Gemma has native JSON support via `format: "json"`
   - Use Pydantic models for schema validation
   - Provide exact field specifications

2. **Few-Shot Examples**
   ```python
   examples = [
       {"input": "class OldAPI:", "output": {"status": "deprecated"}},
       {"input": "// TODO: Remove", "output": {"status": "deprecated"}},
       {"input": ".archive/", "output": {"status": "archived"}}
   ]
   ```

3. **Chain-of-Thought for Complex Classification**
   ```python
   prompt = """
   Step 1: Check file path for archive indicators
   Step 2: Scan imports for frameworks
   Step 3: Count comments vs code ratio
   Step 4: Output JSON with findings
   """
   ```

### Configuration Profiles

```python
class TaggingProfiles:
    """Different configurations for different accuracy needs"""
    
    PRECISE = {
        "temperature": 0.05,   # Ultra-low for maximum consistency
        "top_p": 0.5,         # Very narrow sampling
        "top_k": 5,           # Minimal variation
        "seed": 42,           # Fixed seed
        "num_predict": 1000
    }
    
    BALANCED = {
        "temperature": 0.2,   # Some variation allowed
        "top_p": 0.8,         # Wider sampling
        "top_k": 20,          # More options
        "seed": None,         # Random seed
        "num_predict": 1500
    }
    
    EXPLORATORY = {
        "temperature": 0.4,   # More creative
        "top_p": 0.9,         # Broad sampling
        "top_k": 40,          # Many options
        "seed": None,         # Random seed
        "num_predict": 2000
    }
```

### Validation & Error Handling

```python
class MetadataValidator:
    """Validate LLM outputs for consistency"""
    
    @staticmethod
    def validate_json(response: str) -> Dict:
        """Ensure valid JSON structure"""
        try:
            data = json.loads(response)
            
            # Validate required fields
            required = ["status", "domain", "quality"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate enum values
            if data["status"] not in ["active", "deprecated", "archived", "experimental"]:
                data["status"] = "active"  # Default fallback
            
            return data
        except json.JSONDecodeError:
            # Try to extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise
    
    @staticmethod
    def confidence_check(metadata: FileMetadata) -> bool:
        """Check if confidence is acceptable"""
        return metadata.confidence_score >= 0.7
```

### Performance Optimization

1. **Batch Processing**
   ```python
   async def batch_tag(files: List[Path], batch_size: int = 5):
       """Process files in parallel batches"""
       batches = [files[i:i+batch_size] for i in range(0, len(files), batch_size)]
       for batch in batches:
           await asyncio.gather(*[tag_file(f) for f in batch])
   ```

2. **Response Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_cached_metadata(file_hash: str) -> Optional[FileMetadata]:
       """Cache recent tagging results"""
       return cache.get(file_hash)
   ```

3. **Prompt Template Caching**
   ```python
   PROMPT_TEMPLATES = {
       "python": load_template("python_analysis.txt"),
       "javascript": load_template("js_analysis.txt"),
       "typescript": load_template("ts_analysis.txt")
   }
   ```

## Benefits

### Immediate Benefits
- **Automatic archive detection** - LLM identifies deprecated/archived code
- **Smart filtering** - Search by domain, quality, status
- **Quality insights** - Identify code needing refactoring
- **Technology mapping** - Understand framework usage across codebase
- **Reduced noise** - Filter out low-quality or irrelevant results

### Long-term Benefits
- **Code health monitoring** - Track quality trends over time
- **Dependency analysis** - Understand technology relationships
- **Onboarding aid** - New developers understand codebase structure
- **Technical debt tracking** - Identify areas needing attention
- **Compliance support** - Tag security/regulatory concerns

## Implementation Plan

### Phase 1: Container Setup (Day 1)
1. Add Ollama container to docker-compose
2. Configure Gemma 4B model
3. Test container health and API
4. Benchmark tagging performance

### Phase 2: Tagging Service (Day 2)
1. Implement `MetadataTagger` class
2. Design prompt engineering
3. Add retry and fallback logic
4. Create metadata schema

### Phase 3: Indexer Integration (Day 3)
1. Integrate tagger with indexer
2. Update Qdrant schema for metadata
3. Add async tagging pipeline
4. Handle failures gracefully

### Phase 4: Search Enhancement (Day 4)
1. Add metadata filter parameters
2. Update MCP tool definitions
3. Implement filter UI/API
4. Test end-to-end

### Phase 5: Optimization (Day 5)
1. Tune prompts for accuracy
2. Implement caching layer
3. Add batch processing
4. Performance profiling

## Resource Requirements

### Container Resources
- **Memory**: 6GB (4.4GB model + overhead)
- **CPU**: 2-4 cores recommended
- **Storage**: 10GB for model storage
- **Network**: Internal only (port 51434)

### Performance Targets
- **Tagging latency**: <2s per file
- **Throughput**: 30-50 files/minute
- **Accuracy**: >85% correct classification
- **Availability**: 99% uptime

## Configuration

### Environment Variables
```bash
# .env additions
ENABLE_LLM_TAGGING=true
OLLAMA_HOST=neural-metadata-tagger
OLLAMA_PORT=11434
OLLAMA_MODEL=gemma:4b
TAGGING_BATCH_SIZE=10
TAGGING_TIMEOUT=30
TAGGING_CACHE_TTL=3600
```

### Model Selection
```yaml
# Can switch between models based on needs
models:
  fast: gemma:1b      # Quick, less accurate
  balanced: gemma:4b  # Default, good balance
  accurate: gemma:7b  # More accurate, slower
  code: codegemma:7b  # Specialized for code
```

## Monitoring & Metrics

Track effectiveness with:
- **Tagging rate**: Files tagged per minute
- **Accuracy score**: Manual validation sampling
- **Cache hit rate**: Reuse of tagged metadata
- **Filter usage**: Which metadata filters are most used
- **Quality distribution**: % of code in each quality tier

## Fallback Strategy

When LLM tagging fails:
1. **Pattern-based fallback** - Use file paths and extensions
2. **Historical data** - Reuse previous successful tags
3. **Default classification** - Mark as "unknown" with low confidence
4. **Queue for retry** - Retry failed files during quiet periods

## Privacy & Security

- **Local processing** - No data leaves the container
- **No training** - Model doesn't learn from indexed code  
- **Audit logging** - Track what gets tagged and when
- **Access control** - Container on internal network only

## Alternatives Considered

### 1. Cloud LLM APIs
- **Pros**: More powerful models, no local resources
- **Cons**: Privacy concerns, latency, cost
- **Rejected**: Data sovereignty requirements

### 2. Rule-based Classification
- **Pros**: Fast, deterministic, no resources
- **Cons**: Misses nuance, high maintenance
- **Rejected**: Insufficient for complex codebases

### 3. Hybrid (Rules + LLM)
- **Pros**: Fast path for obvious cases
- **Cons**: Complexity, two systems to maintain
- **Consider**: Future optimization

## Testing Strategy

```python
async def test_metadata_tagging():
    """Test LLM tagging accuracy"""
    
    test_files = [
        ("src/api/server.py", "backend", "active"),
        (".archive/old_api.py", "backend", "archived"),
        ("components/Button.tsx", "ui", "active"),
        ("test/unit/test_auth.py", "test", "active")
    ]
    
    tagger = MetadataTagger()
    
    for file_path, expected_domain, expected_status in test_files:
        content = read_file(file_path)
        metadata = await tagger.tag_code(content, file_path)
        
        assert metadata.domain == expected_domain
        assert metadata.status == expected_status
        assert metadata.confidence_score > 0.7
```

## Future Enhancements

1. **Multi-model ensemble** - Combine multiple LLMs for consensus
2. **Fine-tuning** - Train Gemma on specific codebase patterns
3. **Incremental updates** - Only re-tag changed files
4. **Relationship extraction** - Detect file dependencies
5. **Security scanning** - Identify vulnerabilities
6. **License detection** - Track open source usage

## References

- Gemma 3 4B Model - 128K context, code-trained
- Ollama Container - Production deployment guide
- LangChain Tagging - Classification patterns
- RAGOps 2025 - Metadata for 40% retrieval improvement
- Google Cloud Run - GPU-accelerated Ollama deployment

## Decision Outcome

Deploy the Ollama + Gemma 4B metadata tagging container to automatically classify and enrich code metadata during indexing. This provides intelligent filtering and search capabilities while maintaining data privacy and reasonable resource usage.

**Target: Deploy in 5 days with Phase 1-3 as MVP, 85% classification accuracy**