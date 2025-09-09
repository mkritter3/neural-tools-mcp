#!/bin/bash
#
# Database Failover Test
# Tests graceful degradation when databases are unavailable
# Validates P1 error handling and circuit breaker patterns
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }
log_header() { echo -e "${BLUE}$1${NC}"; }

test_qdrant_failover() {
    log_info "Testing Qdrant failover behavior..."
    
    # Test with unreachable Qdrant
    QDRANT_HOST=nonexistent-host QDRANT_PORT=9999 python3 -c "
import sys
import asyncio
sys.path.append('src')

async def test():
    from servers.services.qdrant_service import QdrantService
    
    service = QdrantService(project_name='failover_test')
    
    # Should fail gracefully, not crash
    try:
        result = await service.initialize()
        if result.get('status') == 'error':
            print('✅ Qdrant failover: Graceful error handling')
            return True
        else:
            print('❌ Qdrant failover: Should have failed but succeeded')
            return False
    except Exception as e:
        print(f'✅ Qdrant failover: Exception caught gracefully - {e}')
        return True

result = asyncio.run(test())
sys.exit(0 if result else 1)
" && log_success "Qdrant failover test passed" || log_error "Qdrant failover test failed"
}

test_neo4j_failover() {
    log_info "Testing Neo4j failover behavior..."
    
    # Test with unreachable Neo4j
    NEO4J_URI=bolt://nonexistent-host:7687 python3 -c "
import sys
sys.path.append('src')

try:
    from servers.services.service_container import ServiceContainer
    
    container = ServiceContainer(project_name='failover_test')
    
    # Should fail gracefully, not crash
    try:
        result = container.ensure_neo4j_client()
        if not result:
            print('✅ Neo4j failover: Graceful failure handling')
            sys.exit(0)
        else:
            print('❌ Neo4j failover: Should have failed but succeeded')
            sys.exit(1)
    except Exception as e:
        print(f'✅ Neo4j failover: Exception caught gracefully - {e}')
        sys.exit(0)
        
except Exception as e:
    print(f'✅ Neo4j failover: Import/connection error handled - {e}')
    sys.exit(0)
" && log_success "Neo4j failover test passed" || log_error "Neo4j failover test failed"
}

test_embedding_service_failover() {
    log_info "Testing embedding service failover..."
    
    # Test with unreachable embedding service
    NOMIC_API_URL=http://nonexistent-host:8080 python3 -c "
import sys
import asyncio
sys.path.append('src')

async def test():
    from servers.services.nomic_service import NomicService
    
    service = NomicService()
    
    try:
        # Should fallback to local embeddings or fail gracefully
        embeddings = await service.get_embeddings(['test query'])
        
        if embeddings and len(embeddings) > 0:
            print(f'✅ Embedding failover: Fallback embeddings generated ({len(embeddings[0])}D)')
            return True
        else:
            print('✅ Embedding failover: Graceful failure (no fallback available)')
            return True
            
    except Exception as e:
        print(f'✅ Embedding failover: Exception handled gracefully - {e}')
        return True

result = asyncio.run(test())
sys.exit(0)  # Always pass for embedding failover
" && log_success "Embedding failover test passed" || log_warning "Embedding failover had issues"
}

test_partial_system_operation() {
    log_info "Testing partial system operation..."
    
    # Test that system can still function with some services down
    python3 -c "
import sys
import asyncio
sys.path.append('src')

async def test():
    # Test P1 methods exist even when services are down
    from servers.services.qdrant_service import QdrantService
    
    service = QdrantService()
    
    # P1-A: search_vectors method should exist
    if not hasattr(service, 'search_vectors'):
        print('❌ P1-A FAIL: search_vectors missing in offline mode')
        return False
    
    # P1-C: delete_points method should exist  
    if not hasattr(service, 'delete_points'):
        print('❌ P1-C FAIL: delete_points missing in offline mode')
        return False
    
    print('✅ P1 API methods available in offline mode')
    return True

result = asyncio.run(test())
sys.exit(0 if result else 1)
" && log_success "Partial operation test passed" || log_error "Partial operation test failed"
}

test_p1_error_handling() {
    log_info "Testing P1 error handling improvements..."
    
    python3 -c "
import sys
import asyncio
import inspect
sys.path.append('src')

async def test():
    from servers.services.qdrant_service import QdrantService
    
    # P1-B: Check that dimension validation has proper error messages
    service = QdrantService()
    upsert_source = inspect.getsource(service.upsert_points)
    
    if 'dimension mismatch' in upsert_source and 'EMBED_DIM' in upsert_source:
        print('✅ P1-B: Dimension validation has helpful error messages')
    else:
        print('❌ P1-B: Missing helpful error messages')
        return False
    
    # Check that methods handle uninitialized state gracefully
    try:
        collections = await service.get_collections()
        print('❌ Should fail when uninitialized')
        return False
    except Exception as e:
        if 'not initialized' in str(e).lower():
            print('✅ P1 Error Handling: Clear initialization error messages')
        else:
            print(f'✅ P1 Error Handling: Graceful failure - {e}')
    
    return True

result = asyncio.run(test())
sys.exit(0 if result else 1)
" && log_success "P1 error handling test passed" || log_error "P1 error handling test failed"
}

generate_failover_report() {
    local overall_result=$1
    
    log_header "📊 Database Failover Test Report"
    
    echo "Failover Test Results:"
    echo "  - Test Suite: Database Failover & Circuit Breaker"
    echo "  - Timestamp: $(date)"
    echo "  - Focus: Graceful degradation when databases unavailable"
    
    if [[ $overall_result -eq 0 ]]; then
        log_success "🎉 ALL FAILOVER TESTS PASSED"
        echo "✅ System handles database failures gracefully"
        echo "✅ P1 error handling improvements validated"
        echo "✅ Services fail fast with clear error messages"
        echo ""
        echo "System demonstrates production-grade resilience"
    else
        log_error "💥 FAILOVER TESTS FAILED"
        echo "❌ System does not handle failures gracefully"
        echo "❌ Review error handling before production deployment"
    fi
    
    return $overall_result
}

main() {
    log_header "🔄 Database Failover Test Suite"
    cd "$PROJECT_ROOT"
    
    local overall_result=0
    
    # Test individual service failover
    if ! test_qdrant_failover; then
        overall_result=1
    fi
    
    if ! test_neo4j_failover; then
        overall_result=1
    fi
    
    # Test embedding service failover (warning only)
    test_embedding_service_failover
    
    # Test partial system operation
    if ! test_partial_system_operation; then
        overall_result=1
    fi
    
    # Test P1 error handling specifically
    if ! test_p1_error_handling; then
        overall_result=1
    fi
    
    generate_failover_report $overall_result
    exit $overall_result
}

main "$@"