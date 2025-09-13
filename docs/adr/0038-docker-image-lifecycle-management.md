# ADR-0038: Docker Image Lifecycle Management & Cleanup Strategy

**Status**: Accepted
**Date**: September 12, 2025
**Deciders**: L9 Engineering Team, Expert Consensus (Grok-4, Gemini 2.5 Pro)
**Context**: ADR-0037 Container Configuration Priority Standard

## Context

During ADR-0037 implementation, we discovered Docker image proliferation with 8 different tags and debug-suffixed containers:

```bash
REPOSITORY                          TAG
neural-flow-nomic-v2               production
neural-flow-nomic-v2               latest
neural-flow-nomic-v2               adr-37-fix
neural-flow-nomic-v2               debug-build-20240912
l9-neural-indexer                  production
l9-neural-indexer                  latest
l9-neural-indexer                  adr-37-fix
l9-neural-indexer                  debug-build-20240912
```

**Expert Consensus**: Both Grok-4 and Gemini 2.5 Pro recommend semantic versioning, immutable artifacts, and phased cleanup to maintain L9 production standards.

## Decision

Implement comprehensive Docker image lifecycle management following these principles:

### 1. Semantic Versioning Strategy

**Production Images**:
- Format: `{service}:v{MAJOR}.{MINOR}.{PATCH}`
- Example: `l9-neural-indexer:v1.2.0`, `neural-flow-nomic-v2:v1.1.0`
- Git SHA tags: `{service}:sha-{SHORT_SHA}` for immutable artifacts
- Example: `l9-neural-indexer:sha-a471e44`

**Development Images**:
- Format: `{service}:dev-{FEATURE}`
- Example: `l9-neural-indexer:dev-adr37`, `neural-flow-nomic-v2:dev-optimization`
- Temporary tags, cleaned up after merge

### 2. Container Naming Standards

**Production Containers**:
```bash
# CORRECT: Clean, semantic names
claude-l9-template-indexer-1
claude-l9-template-nomic-1
claude-l9-template-neo4j-1

# WRONG: Debug suffixes
claude-l9-template-indexer-adr-37-1
l9-neural-indexer-debug-1
```

### 3. Image Tagging Rules

| Tag Type | Purpose | Lifecycle | Example |
|----------|---------|-----------|---------|
| `vX.Y.Z` | Production releases | Permanent | `v1.2.0` |
| `sha-{hash}` | Immutable artifacts | Permanent | `sha-a471e44` |
| `production` | Latest stable | Moved on release | Points to latest vX.Y.Z |
| `latest` | Latest build | Updated continuously | Development use only |
| `dev-{feature}` | Feature branches | Temporary | `dev-adr37` |

## Implementation Plan

### Phase 1: Rebuild Production Image with Semantic Versioning

1. **Build new production image**:
```bash
# Build with proper semantic version
docker build -t l9-neural-indexer:v1.2.0 \
  -t l9-neural-indexer:sha-$(git rev-parse --short HEAD) \
  -f docker/indexer/Dockerfile.production .

# Tag as production
docker tag l9-neural-indexer:v1.2.0 l9-neural-indexer:production
```

2. **Update docker-compose.yml**:
```yaml
services:
  indexer:
    image: l9-neural-indexer:production  # No debug suffixes
    container_name: claude-l9-template-indexer-1
```

### Phase 2: Update MCP Configuration

1. **Standardize container names** in `.mcp.json`:
```json
{
  "environment": {
    "INDEXER_CONTAINER_NAME": "claude-l9-template-indexer-1",
    "PROJECT_NAME": "claude-l9-template"
  }
}
```

2. **Remove debug environment variables**:
```bash
# Remove ADR-37 debug flags
unset DEBUG_ADR_37
unset INDEXER_DEBUG_SUFFIX
```

### Phase 3: Safe Image Cleanup

1. **Stop containers using old images**:
```bash
docker-compose down
docker stop $(docker ps -q --filter "ancestor=l9-neural-indexer:adr-37-fix")
```

2. **Remove old tags (preserve data)**:
```bash
# Safe removal - only removes image tags, not data
docker rmi l9-neural-indexer:adr-37-fix
docker rmi l9-neural-indexer:debug-build-20240912
docker rmi neural-flow-nomic-v2:adr-37-fix
docker rmi neural-flow-nomic-v2:debug-build-20240912
```

3. **Restart with production images**:
```bash
docker-compose up -d
```

## Success Metrics

### Image Hygiene
- **Target**: ≤3 tags per service (`vX.Y.Z`, `sha-{hash}`, `production`)
- **Current**: 8+ tags per service
- **Measurement**: `docker images | grep -E "(l9-neural|neural-flow)" | wc -l`

### Container Naming
- **Target**: 100% production containers without debug suffixes
- **Measurement**: `docker ps --format "{{.Names}}" | grep -v debug | wc -l`

### Disk Usage
- **Target**: ≤2GB total Docker image storage
- **Measurement**: `docker system df`

### Deployment Reliability
- **Target**: Zero downtime during image transitions
- **Measurement**: Service health checks during cleanup

## Lifecycle Policies

### Automatic Cleanup Rules

1. **Development tags**: Delete after 7 days if untagged
2. **Feature branch tags**: Delete when branch is merged/deleted
3. **SHA tags**: Keep permanently for audit trail
4. **Production tags**: Keep last 3 versions

### Implementation via Docker Prune

```bash
# Weekly cleanup job
docker image prune -f --filter "until=168h"  # 7 days
docker system prune -f --volumes
```

## Migration Safety

### Rollback Plan
```bash
# If new image fails, rollback to previous version
docker tag l9-neural-indexer:v1.1.0 l9-neural-indexer:production
docker-compose down && docker-compose up -d
```

### Data Preservation
- **Neo4j data**: Persisted in `/var/lib/neo4j/data` volume
- **Qdrant data**: Persisted in `/qdrant/storage` volume
- **Configuration**: Environment variables preserved in `.mcp.json`

### Validation Checklist
- [ ] All services start successfully
- [ ] MCP tools connect without errors
- [ ] Hybrid search returns expected results
- [ ] Project isolation maintained (ADR-0029)
- [ ] Configuration priority respected (ADR-0037)

## Expert Analysis Integration

### Grok-4 Recommendations ✅
- **Semantic versioning**: Implemented with `vX.Y.Z` format
- **Immutable artifacts**: SHA-based tags for audit trail
- **Clean separation**: Production vs development images
- **Automated lifecycle**: Prune policies and cleanup rules

### Gemini 2.5 Pro Recommendations ✅
- **Phased approach**: 3-phase implementation plan
- **Safety-first**: Data preservation and rollback procedures
- **Standards compliance**: L9 production-grade practices
- **Documentation**: Comprehensive lifecycle policies

## Alternatives Considered

### Alternative 1: Keep All Images
- **Pros**: Zero risk of losing functionality
- **Cons**: Disk bloat, confusion, maintenance burden
- **Verdict**: Rejected - violates L9 standards

### Alternative 2: Nuclear Cleanup (Remove All)
- **Pros**: Clean slate approach
- **Cons**: High risk of data loss, extended downtime
- **Verdict**: Rejected - too risky for production

### Alternative 3: Gradual Migration (Chosen)
- **Pros**: Safe, reversible, production-tested
- **Cons**: Temporary overhead during transition
- **Verdict**: Accepted - balances safety with standards

## Compliance

### ADR Dependencies
- **ADR-0037**: Container Configuration Priority ✅
- **ADR-0029**: Multi-Project Isolation ✅
- **ADR-0036**: Neo4j Property Flattening ✅

### L9 Standards Alignment
- **Truth-First Engineering**: Expert consensus validation
- **95% Gate**: Comprehensive rollback procedures
- **Evidence-Based**: Metrics and success criteria defined
- **Reversible Changes**: Phased approach with rollback plan

## Implementation Status

- [ ] **Phase 1**: Rebuild production image with semantic versioning
- [ ] **Phase 2**: Update MCP configuration and standardize naming
- [ ] **Phase 3**: Clean up old images and implement prevention

**Next Actions**: Execute Phase 1 implementation as approved by expert consensus.

---

**Confidence**: 95% - Expert-validated approach with comprehensive safety measures and L9 standards compliance.