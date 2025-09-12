# ADR-0025: Hybrid Metadata Extraction Strategy

**Date**: September 11, 2025
**Status**: Deferred
**Decision**: Keep pattern-based extraction only (LLM approach deferred due to platform constraints)

## Context

We successfully implemented pattern-based metadata extraction for code (ADR-0023/0024), achieving:
- 100x faster extraction than LLM approach
- Deterministic, reliable results
- Zero timeouts or performance issues
- Rich metadata including dependencies, public APIs, type hints, TODOs, I/O operations

However, we identified that narrative/creative text requires semantic understanding that patterns cannot provide:
- Character relationships
- Emotional tone
- Plot elements
- Themes and symbolism

## Decision

Implement a **hybrid dual-path extraction system**:

1. **KEEP existing pattern-based extraction for code** (working perfectly)
2. **ADD optional vLLM-based extraction for narrative text** (new capability)
3. **Route based on file type/content type**

## Architecture

```python
class MetadataExtractor:
    def __init__(self):
        self.pattern_extractor = PatternBasedExtractor()  # Current working system
        self.llm_extractor = VLLMExtractor()  # New for narrative
        
    async def extract(self, content: str, file_path: str) -> FileMetadata:
        content_type = self.detect_content_type(file_path, content)
        
        if content_type == 'code':
            # Use fast pattern extraction (current system)
            return await self.pattern_extractor.extract(content, file_path)
        elif content_type == 'narrative':
            # Use vLLM for semantic understanding
            return await self.llm_extractor.extract(content, file_path)
        else:
            # Default to patterns for unknown types
            return await self.pattern_extractor.extract(content, file_path)
```

## vLLM Integration Design

### Container Service
```yaml
# Add to docker-compose.yml
vllm-metadata:
  image: vllm/vllm-openai:latest
  container_name: neural-vllm-metadata
  ports:
    - "48002:8000"
  volumes:
    - vllm_models:/root/.cache/huggingface
  command: >
    --model Qwen/Qwen2.5-0.5B-Instruct
    --dtype float16
    --max-model-len 2048
    --guided-decoding-backend outlines
    --gpu-memory-utilization 0.5
  deploy:
    resources:
      limits:
        memory: 4G
  restart: unless-stopped
```

### Direct HTTP Client (No OpenAI SDK)
```python
class VLLMExtractor:
    def __init__(self, base_url="http://localhost:48002"):
        self.base_url = base_url
        
    async def extract(self, content: str, file_path: str) -> FileMetadata:
        async with aiohttp.ClientSession() as session:
            # Direct vLLM API call
            response = await session.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "prompt": self._build_prompt(content),
                    "max_tokens": 200,
                    "temperature": 0.1,
                    "guided_json": self._get_schema()  # JSON schema enforcement
                }
            )
            # Parse structured response
            return self._parse_response(await response.json())
```

## Content Type Detection

```python
def detect_content_type(self, file_path: str, content: str) -> str:
    # Code indicators
    if any(file_path.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.go']):
        return 'code'
    
    # Narrative indicators
    if any(file_path.endswith(ext) for ext in ['.md', '.txt', '.story', '.chapter']):
        # Check content for narrative patterns
        if any(word in content.lower() for word in ['chapter', 'scene', 'character', 'dialogue']):
            return 'narrative'
    
    # Check content structure
    code_patterns = ['def ', 'class ', 'import ', 'function', 'const ', 'var ']
    if any(pattern in content for pattern in code_patterns):
        return 'code'
    
    return 'unknown'  # Default to pattern extraction
```

## Safety Measures

1. **Fallback Strategy**: If vLLM times out (>5s), fall back to pattern extraction
2. **Circuit Breaker**: After 3 failures, disable LLM path for 60 seconds
3. **Separate Queues**: Keep code and narrative in separate processing queues
4. **Feature Flag**: Environment variable to disable LLM path entirely

```python
ENABLE_LLM_EXTRACTION = os.getenv('ENABLE_LLM_EXTRACTION', 'false')
```

## Migration Path

1. **Phase 1**: Keep current system unchanged, add vLLM container
2. **Phase 2**: Implement VLLMExtractor class with same interface
3. **Phase 3**: Add content type detection
4. **Phase 4**: Enable for `.md` files only (low risk)
5. **Phase 5**: Expand to other narrative formats

## Benefits

- **No Breaking Changes**: Current code extraction continues working
- **Progressive Enhancement**: Add capabilities without risk
- **Best Tool for Job**: Patterns for code, LLM for narrative
- **Performance**: Code stays fast, narrative gets semantic understanding
- **Flexibility**: Easy to disable/enable per content type

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| vLLM container fails | Pattern extraction continues working |
| Slow LLM responses | 5-second timeout with fallback |
| Memory pressure | Separate container with resource limits |
| Wrong content detection | Defaults to safe pattern extraction |

## Alternatives Considered

1. **Use LLM for everything**: Too slow, we already proved this
2. **Patterns for everything**: Can't understand narrative semantics
3. **Replace Ollama entirely**: Would break existing setup
4. **Cloud LLM API**: Latency and cost concerns

## Implementation Checklist

- [ ] Add vLLM service to docker-compose.yml
- [ ] Create VLLMExtractor class
- [ ] Implement content type detection
- [ ] Add fallback mechanisms
- [ ] Test with sample narrative files
- [ ] Verify code extraction still works
- [ ] Add feature flag for production safety

## Success Metrics

- Code extraction remains <10ms per file
- Narrative extraction completes <2s per file
- Zero impact on existing code pipeline
- 95% success rate for narrative metadata
- Graceful degradation on failures

## Decision

Proceed with hybrid approach - patterns for code, vLLM for narrative text.

---

**Confidence: 95%**
Assumptions: vLLM provides structured generation, content type detection is reliable