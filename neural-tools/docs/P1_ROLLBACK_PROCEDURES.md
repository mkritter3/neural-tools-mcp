# P1 Rollback Procedures
**Emergency rollback guide for P1 integration fixes**

## Quick Rollback Commands

```bash
# 1. Revert API method changes
git checkout HEAD~1 -- src/infrastructure/quantization.py
git checkout HEAD~1 -- src/servers/tools/core_tools.py

# 2. Revert dimension validation
git checkout HEAD~1 -- src/servers/services/qdrant_service.py
git checkout HEAD~1 -- src/servers/services/service_container.py

# 3. Revert configuration changes
git checkout HEAD~1 -- docker-compose.yml
rm -rf src/servers/config/

# 4. Restart services
docker-compose restart
```

## Rollback Risk Assessment

### P1-A: API Method Changes (LOW RISK)
- **Files**: `quantization.py:782`, `core_tools.py:64`  
- **Rollback**: Change `.search_vectors()` back to `.search()`
- **Impact**: MCP semantic search may fail with "method not found"
- **Detection**: Search requests return method errors

### P1-B: Dimension Validation (MEDIUM RISK)
- **Files**: `qdrant_service.py:185-200`, `service_container.py:82`
- **Rollback**: Remove dimension validation, revert to hardcoded `size=768`
- **Impact**: Mismatched vectors may be accepted silently
- **Detection**: No immediate failure, but search quality degrades

### P1-C: Delete Points Method (LOW RISK)  
- **Files**: `qdrant_service.py:480-534`
- **Rollback**: Remove `delete_points()` method entirely
- **Impact**: Deletion operations fail with "method not found"
- **Detection**: Calls to delete_points raise AttributeError

### P1-D: Centralized Configuration (HIGH RISK)
- **Files**: `src/servers/config/runtime.py`, `docker-compose.yml:119`
- **Rollback**: Remove config module, restore `QDRANT_HTTP_PORT=6333`
- **Impact**: Service discovery may fail, connection errors
- **Detection**: Services fail to start, port binding issues

## Staged Rollback Strategy

### Level 1: Non-Breaking Changes First
```bash
# Remove new delete_points method (least risky)
git checkout HEAD~1 -- src/servers/services/qdrant_service.py
# Keep: lines 175-210 (upsert dimension validation)
# Remove: lines 480-534 (delete_points method)
```

### Level 2: API Compatibility  
```bash
# Revert API method names if clients are failing
git checkout HEAD~1 -- src/infrastructure/quantization.py
git checkout HEAD~1 -- src/servers/tools/core_tools.py
```

### Level 3: Full Configuration Rollback
```bash
# Only if service discovery is broken
rm -rf src/servers/config/
git checkout HEAD~1 -- docker-compose.yml
git checkout HEAD~1 -- src/servers/services/service_container.py
```

## Validation After Rollback

```bash
# 1. Test MCP server starts
python3 -m src.neural_mcp.neural_server_stdio &
sleep 3 && kill $!

# 2. Test database connections
docker-compose exec neural-tools python3 -c "
from servers.services.service_container import ServiceContainer
container = ServiceContainer('test')
print('Qdrant:', container.ensure_qdrant_client())
print('Neo4j:', container.ensure_neo4j_client())
"

# 3. Test semantic search
python3 -c "
from servers.tools.core_tools import semantic_code_search
result = semantic_code_search('test query')
print('Search working:', 'results' in result)
"
```

## Monitoring During Rollback

- **Qdrant Errors**: `docker logs claude-l9-template-qdrant-1`
- **Neo4j Errors**: `docker logs claude-l9-template-neo4j-1` 
- **MCP Errors**: Check for `AttributeError`, `ConnectionRefused`, `TypeError`
- **Dimension Mismatches**: Look for vector size errors in logs

## Emergency Contacts

- **L9 Engineer**: Available for rollback assistance
- **Database Admin**: For Qdrant/Neo4j connection issues
- **MCP Client Teams**: Notify of API method changes

## Post-Rollback Actions

1. **Update incident report** with rollback timeline
2. **Validate all dependent services** are working
3. **Plan gradual re-deployment** with additional safeguards  
4. **Review P1 changes** for safer implementation approach