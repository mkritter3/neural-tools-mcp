# ADR-0006: Neo4j GraphRAG Migration for L9 Neural Tools System

**Status:** Proposed  
**Date:** 2025-08-30  
**Authors:** Claude L9 Engineering Team  
**Reviewers:** TBD  
**Approvers:** TBD  

## Summary

Migrate from Kuzu embedded graph database to Neo4j client-server architecture to resolve critical concurrency and multi-tenancy issues in the L9 Neural Tools system, ensuring production-grade reliability for large codebase GraphRAG operations.

## Context and Problem Statement

### Current Architecture Issues

The L9 Neural Tools system currently uses Kuzu as an embedded graph database for GraphRAG (Graph Retrieval-Augmented Generation) functionality. Through systematic analysis and AI expert consensus (Grok-4: 8/10 confidence, Gemini-2.5-Pro: 9/10 confidence), we have identified fundamental architecture incompatibilities:

**Critical Problems:**
1. **File Locking Conflicts**: `IO exception: Could not set lock on file : /app/kuzu/graph.db`
2. **Multi-tenancy Failures**: Kuzu's single-writer/multiple-reader model prevents concurrent project access
3. **Container Restart Issues**: Stale locks from ungraceful shutdowns block system recovery
4. **Scalability Ceiling**: Cannot reliably run multiple projects simultaneously

### Expert Consensus Analysis

**Research Date:** August 30, 2025  
**Sources:** Kuzu v0.6.0+ documentation, Neo4j 5.15 documentation, industry best practices

**Key Findings:**
- Kuzu is an embedded database (like SQLite), not designed for service architectures
- Neo4j is the industry standard for multi-tenant graph operations
- Current approach is a documented anti-pattern for containerized systems
- Migration complexity: Low-Medium (estimated 2-4 days)

## Goals and Success Criteria

### Primary Goals
1. **Eliminate Concurrency Issues**: Zero file locking errors in production
2. **Enable Multi-tenancy**: Support concurrent access across multiple projects  
3. **Maintain GraphRAG Functionality**: Preserve all code relationship mapping capabilities
4. **Improve System Reliability**: Achieve 99.9% uptime for graph operations
5. **Production-grade Architecture**: Align with industry best practices

### Success Metrics
- **Reliability**: Zero lock-related failures over 30-day period
- **Performance**: Graph queries ≤ 100ms for 95th percentile
- **Scalability**: Support 10+ concurrent projects without degradation
- **Data Integrity**: 100% data consistency across container restarts
- **Developer Experience**: Same or improved API interface

## Technical Decision

### Solution Architecture

**Selected Approach:** Neo4j Community Edition 5.15 with Docker containerization

**Rationale:**
- **Industry Standard**: Neo4j created Cypher query language, extensive ecosystem
- **Production Ready**: ACID transactions, proven concurrency model
- **Multi-tenancy**: Native support for multiple databases per instance
- **Docker Native**: Official Docker images with orchestration support
- **Cost Effective**: Community edition meets all L9 requirements
- **Migration Path**: Minimal code changes (same Cypher language)

### Alternative Analysis

| **Option** | **Pros** | **Cons** | **L9 Suitability** |
|---|---|---|---|
| **Keep Kuzu + Isolation** | Minimal changes | Brittle, maintenance burden | ❌ Not L9-grade |
| **Neo4j Community** | Production-ready, free | Learning curve | ✅ **SELECTED** |
| **Neo4j Enterprise** | Advanced features | Licensing costs | ⚠️ Overkill for current needs |
| **ArangoDB** | Multi-model database | Less GraphRAG ecosystem | 🟡 Alternative option |
| **MemGraph** | High performance | Smaller community | 🟡 Alternative option |

## Implementation Plan

### Phase 1: Foundation (Days 1-2)

**Objectives:**
- Set up Neo4j infrastructure
- Verify core connectivity
- Establish data migration strategy

**Tasks:**
1. **Docker Infrastructure Setup**
   ```yaml
   # Add to docker-compose.neural-tools.yml
   neo4j-graphrag:
     image: neo4j:5.15-community
     container_name: ${PROJECT_NAME}-neo4j
     environment:
       - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
       - NEO4J_PLUGINS=["apoc", "graph-data-science"]
       - NEO4J_dbms_default__database=neo4j
     ports:
       - "${NEO4J_HTTP_PORT:-7474}:7474"
       - "${NEO4J_BOLT_PORT:-7687}:7687"
     volumes:
       - ./.neural-tools/neo4j/${PROJECT_NAME}/data:/data
       - ./.neural-tools/neo4j/${PROJECT_NAME}/logs:/logs
     networks:
       - neural-network
     healthcheck:
       test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
       interval: 30s
       timeout: 10s
       retries: 3
   ```

2. **Python Dependencies Update**
   ```txt
   # requirements-neo4j.txt
   neo4j>=5.15.0,<6.0.0
   py2neo>=2021.2.4
   neotime>=1.7.4
   ```

3. **Neo4j Client Setup**
   ```python
   # neo4j_client.py
   from neo4j import GraphDatabase
   from typing import Dict, List, Any
   import os

   class Neo4jGraphClient:
       def __init__(self):
           uri = os.getenv("NEO4J_URI", "bolt://neo4j-graphrag:7687")
           username = os.getenv("NEO4J_USERNAME", "neo4j")
           password = os.getenv("NEO4J_PASSWORD")
           self.driver = GraphDatabase.driver(uri, auth=(username, password))
           self.database = os.getenv("NEO4J_DATABASE", f"{PROJECT_NAME}_graph")
       
       def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
           with self.driver.session(database=self.database) as session:
               result = session.run(query, parameters or {})
               return [record.data() for record in result]
       
       def close(self):
           self.driver.close()
   ```

**Exit Criteria for Phase 1:**
- [ ] Neo4j container starts successfully
- [ ] Health check passes consistently
- [ ] Python client connects without errors
- [ ] Basic Cypher query executes: `RETURN 1 AS test`
- [ ] Project-specific database creation works

### Phase 2: Data Migration (Days 2-3)

**Objectives:**
- Migrate existing Kuzu data to Neo4j
- Establish data consistency verification
- Create rollback procedures

**Tasks:**

1. **Schema Migration Script**
   ```python
   # migrate_kuzu_to_neo4j.py
   import asyncio
   from kuzu import Database as KuzuDB, Connection as KuzuConn
   from neo4j_client import Neo4jGraphClient
   import logging

   class GraphMigrator:
       def __init__(self, kuzu_path: str, neo4j_client: Neo4jGraphClient):
           self.kuzu_db = KuzuDB(kuzu_path) if os.path.exists(kuzu_path) else None
           self.kuzu_conn = KuzuConn(self.kuzu_db) if self.kuzu_db else None
           self.neo4j_client = neo4j_client
           
       async def migrate_schema(self):
           """Migrate node and relationship schemas"""
           if not self.kuzu_conn:
               logging.info("No Kuzu data to migrate")
               return
               
           # Create constraints and indexes
           schema_queries = [
               "CREATE CONSTRAINT file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.path IS UNIQUE",
               "CREATE CONSTRAINT function_id_unique IF NOT EXISTS FOR (fn:Function) REQUIRE fn.id IS UNIQUE", 
               "CREATE INDEX function_name_index IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
               "CREATE INDEX file_extension_index IF NOT EXISTS FOR (f:File) ON (f.extension)",
           ]
           
           for query in schema_queries:
               self.neo4j_client.execute_query(query)
               
       async def migrate_nodes(self):
           """Migrate all nodes from Kuzu to Neo4j"""
           # Get all node tables from Kuzu
           tables_result = self.kuzu_conn.execute("CALL SHOW_TABLES() RETURN *")
           
           for table in tables_result:
               table_name = table[0]
               logging.info(f"Migrating table: {table_name}")
               
               # Read all data from Kuzu table
               kuzu_result = self.kuzu_conn.execute(f"MATCH (n:{table_name}) RETURN n")
               
               # Batch insert into Neo4j
               batch_size = 1000
               nodes = []
               
               for record in kuzu_result:
                   node_data = record[0]
                   nodes.append(node_data)
                   
                   if len(nodes) >= batch_size:
                       await self._batch_insert_nodes(table_name, nodes)
                       nodes = []
               
               # Insert remaining nodes
               if nodes:
                   await self._batch_insert_nodes(table_name, nodes)
       
       async def _batch_insert_nodes(self, label: str, nodes: List[Dict]):
           """Batch insert nodes into Neo4j"""
           query = f"""
           UNWIND $nodes AS nodeData
           CREATE (n:{label})
           SET n = nodeData
           """
           self.neo4j_client.execute_query(query, {"nodes": nodes})

   # Migration execution
   async def run_migration():
       neo4j_client = Neo4jGraphClient()
       kuzu_path = os.getenv("KUZU_DB_PATH", "/app/kuzu")
       
       migrator = GraphMigrator(kuzu_path, neo4j_client)
       await migrator.migrate_schema()
       await migrator.migrate_nodes()
       await migrator.migrate_relationships()
       
       # Verify migration
       verification_result = await verify_migration_integrity(migrator)
       
       if verification_result['success']:
           logging.info("✅ Migration completed successfully")
           return True
       else:
           logging.error(f"❌ Migration failed: {verification_result['errors']}")
           return False
   ```

2. **Data Integrity Verification**
   ```python
   async def verify_migration_integrity(migrator: GraphMigrator) -> Dict[str, Any]:
       """Verify data was migrated correctly"""
       verification_results = {
           'success': True,
           'errors': [],
           'stats': {}
       }
       
       try:
           # Compare node counts
           kuzu_node_count = migrator.kuzu_conn.execute("MATCH (n) RETURN COUNT(n) AS count")[0][0]
           neo4j_node_count = migrator.neo4j_client.execute_query("MATCH (n) RETURN COUNT(n) AS count")[0]['count']
           
           verification_results['stats']['kuzu_nodes'] = kuzu_node_count
           verification_results['stats']['neo4j_nodes'] = neo4j_node_count
           
           if kuzu_node_count != neo4j_node_count:
               verification_results['success'] = False
               verification_results['errors'].append(f"Node count mismatch: Kuzu={kuzu_node_count}, Neo4j={neo4j_node_count}")
           
           # Sample data verification
           sample_queries = [
               "MATCH (f:File) RETURN COUNT(f) AS file_count",
               "MATCH (fn:Function) RETURN COUNT(fn) AS function_count", 
               "MATCH (c:Class) RETURN COUNT(c) AS class_count"
           ]
           
           for query in sample_queries:
               kuzu_result = migrator.kuzu_conn.execute(query)
               neo4j_result = migrator.neo4j_client.execute_query(query)
               # Add verification logic
               
       except Exception as e:
           verification_results['success'] = False
           verification_results['errors'].append(str(e))
       
       return verification_results
   ```

**Exit Criteria for Phase 2:**
- [ ] All Kuzu nodes migrated to Neo4j (100% count match)
- [ ] All relationships preserved with correct cardinality  
- [ ] Sample queries return identical results
- [ ] Data integrity verification passes
- [ ] Rollback procedure tested and documented

### Phase 3: Code Integration (Days 3-4)

**Objectives:**
- Replace Kuzu client with Neo4j client
- Update all GraphRAG functions
- Maintain API compatibility

**Tasks:**

1. **MCP Server Integration**
   ```python
   # Update neural-mcp-server-enhanced.py
   
   # Replace Kuzu imports
   # from kuzu import Database, Connection  # REMOVE
   from neo4j_client import Neo4jGraphClient  # ADD
   
   # Initialize Neo4j client (replaces Kuzu)
   try:
       neo4j_client = Neo4jGraphClient()
       GRAPHRAG_ENABLED = True
       logger.info("✅ Neo4j GraphRAG client initialized")
   except Exception as e:
       neo4j_client = None
       GRAPHRAG_ENABLED = False
       logger.error(f"❌ Neo4j GraphRAG client failed: {e}")
   
   # Update kuzu_graph_query tool
   @mcp.tool()
   async def neo4j_graph_query(query: str) -> Dict[str, Any]:
       """Execute Cypher query on Neo4j graph database
       
       Args:
           query: Cypher query to execute
           
       Returns:
           Query results with metadata
       """
       try:
           if not GRAPHRAG_ENABLED or not neo4j_client:
               return {
                   "status": "error",
                   "message": "Neo4j GraphRAG not available"
               }
           
           # Execute query
           results = neo4j_client.execute_query(query)
           
           return {
               "status": "success",
               "results": results,
               "count": len(results),
               "query": query,
               "database": neo4j_client.database
           }
           
       except Exception as e:
           logger.error(f"Neo4j query failed: {e}")
           return {
               "status": "error", 
               "message": str(e),
               "query": query
           }
   ```

2. **GraphRAG Functions Update**
   ```python
   # Update all graph relationship functions
   async def create_code_relationships(file_path: str, ast_data: Dict):
       """Create code relationships in Neo4j"""
       if not GRAPHRAG_ENABLED or not neo4j_client:
           return
           
       try:
           # Create file node
           neo4j_client.execute_query("""
               MERGE (f:File {path: $file_path})
               SET f.language = $language,
                   f.size_bytes = $size,
                   f.last_modified = datetime()
           """, {
               "file_path": file_path,
               "language": ast_data.get("language"),
               "size": ast_data.get("size_bytes", 0)
           })
           
           # Create function nodes and relationships
           for function in ast_data.get("functions", []):
               neo4j_client.execute_query("""
                   MATCH (f:File {path: $file_path})
                   MERGE (fn:Function {id: $func_id})
                   SET fn.name = $func_name,
                       fn.line_start = $line_start,
                       fn.line_end = $line_end,
                       fn.complexity = $complexity
                   MERGE (f)-[:CONTAINS]->(fn)
               """, {
                   "file_path": file_path,
                   "func_id": f"{file_path}:{function['name']}:{function['line_start']}",
                   "func_name": function["name"],
                   "line_start": function["line_start"],
                   "line_end": function["line_end"],
                   "complexity": function.get("complexity", 1)
               })
           
           # Create call relationships
           for call in ast_data.get("function_calls", []):
               neo4j_client.execute_query("""
                   MATCH (caller:Function {id: $caller_id})
                   MATCH (callee:Function {name: $callee_name})
                   MERGE (caller)-[:CALLS {line: $call_line}]->(callee)
               """, {
                   "caller_id": call["caller_id"],
                   "callee_name": call["callee_name"], 
                   "call_line": call["line"]
               })
               
       except Exception as e:
           logger.error(f"Failed to create code relationships: {e}")
   ```

**Exit Criteria for Phase 3:**
- [ ] All Kuzu references removed from codebase
- [ ] Neo4j client integrated successfully  
- [ ] All MCP tools using Neo4j work correctly
- [ ] GraphRAG functions maintain same output format
- [ ] No regression in existing functionality

### Phase 4: Testing and Validation (Day 4)

**Objectives:**
- Comprehensive testing of Neo4j integration
- Performance benchmarking
- Multi-tenancy validation

**Tasks:**

1. **Unit Tests**
   ```python
   # test_neo4j_integration.py
   import pytest
   import asyncio
   from neo4j_client import Neo4jGraphClient
   from neural_mcp_server_enhanced import neo4j_graph_query, create_code_relationships

   @pytest.fixture
   async def neo4j_client():
       client = Neo4jGraphClient()
       # Setup test database
       client.execute_query("MATCH (n) DETACH DELETE n")  # Clean test DB
       yield client
       client.close()

   @pytest.mark.asyncio
   async def test_basic_connectivity(neo4j_client):
       """Test basic Neo4j connectivity"""
       result = neo4j_client.execute_query("RETURN 1 AS test")
       assert result[0]['test'] == 1

   @pytest.mark.asyncio  
   async def test_graph_query_tool():
       """Test MCP graph query tool"""
       result = await neo4j_graph_query("RETURN 1 AS test")
       assert result['status'] == 'success'
       assert result['results'][0]['test'] == 1

   @pytest.mark.asyncio
   async def test_code_relationship_creation(neo4j_client):
       """Test code relationship creation"""
       ast_data = {
           "language": "python",
           "size_bytes": 1500,
           "functions": [
               {
                   "name": "test_function",
                   "line_start": 10,
                   "line_end": 20,
                   "complexity": 3
               }
           ]
       }
       
       await create_code_relationships("/test/file.py", ast_data)
       
       # Verify function was created
       result = neo4j_client.execute_query("""
           MATCH (f:File {path: "/test/file.py"})-[:CONTAINS]->(fn:Function)
           RETURN fn.name AS name
       """)
       
       assert len(result) == 1
       assert result[0]['name'] == 'test_function'

   @pytest.mark.asyncio
   async def test_multi_tenancy():
       """Test project isolation"""
       # Test with different PROJECT_NAME values
       # Verify data isolation between projects
       pass
   ```

2. **Integration Tests**
   ```python
   # test_neo4j_performance.py
   import time
   import statistics
   from neo4j_client import Neo4jGraphClient

   async def benchmark_neo4j_performance():
       """Benchmark Neo4j query performance"""
       client = Neo4jGraphClient()
       
       # Test queries with timing
       test_queries = [
           "MATCH (f:File) RETURN COUNT(f) AS file_count",
           "MATCH (fn:Function)-[:CALLS]->(called:Function) RETURN fn.name, called.name LIMIT 100",
           "MATCH path = (f:File)-[:CONTAINS*2..4]->(end) RETURN LENGTH(path) AS depth LIMIT 50"
       ]
       
       results = {}
       
       for query in test_queries:
           times = []
           for _ in range(10):  # Run 10 times
               start = time.perf_counter()
               client.execute_query(query)
               end = time.perf_counter()
               times.append((end - start) * 1000)  # Convert to ms
           
           results[query] = {
               'mean_ms': statistics.mean(times),
               'p95_ms': statistics.quantiles(times, n=20)[18],  # 95th percentile
               'p99_ms': statistics.quantiles(times, n=100)[98]  # 99th percentile
           }
       
       # Assert performance requirements
       for query, metrics in results.items():
           assert metrics['p95_ms'] < 100, f"Query too slow: {query} = {metrics['p95_ms']}ms"
       
       client.close()
       return results
   ```

3. **Load Testing**
   ```python
   # test_neo4j_concurrent_access.py
   import asyncio
   import concurrent.futures
   from neo4j_client import Neo4jGraphClient

   async def test_concurrent_project_access():
       """Test multiple projects accessing Neo4j simultaneously"""
       
       async def simulate_project_workload(project_name: str, iterations: int):
           # Simulate different PROJECT_NAME
           os.environ['PROJECT_NAME'] = project_name
           client = Neo4jGraphClient()
           
           for i in range(iterations):
               # Simulate typical GraphRAG operations
               await create_code_relationships(f"/{project_name}/file_{i}.py", sample_ast_data)
               result = await neo4j_graph_query(f"MATCH (f:File) WHERE f.path CONTAINS '{project_name}' RETURN COUNT(f)")
               assert result['status'] == 'success'
           
           client.close()
       
       # Run 5 concurrent projects
       projects = ['project_a', 'project_b', 'project_c', 'project_d', 'project_e']
       tasks = [simulate_project_workload(proj, 20) for proj in projects]
       
       # All should complete without locking errors
       await asyncio.gather(*tasks)
   ```

**Exit Criteria for Phase 4:**
- [ ] All unit tests pass (100% success rate)
- [ ] Integration tests complete successfully
- [ ] Performance benchmarks meet requirements (95th percentile < 100ms)
- [ ] Load testing shows no concurrency issues  
- [ ] Multi-tenancy isolation verified
- [ ] Zero locking errors during stress testing

### Phase 5: Deployment and Monitoring (Day 4-5)

**Objectives:**
- Deploy Neo4j to production environment
- Establish monitoring and alerting
- Document operational procedures

**Tasks:**

1. **Production Deployment Script**
   ```bash
   #!/bin/bash
   # deploy_neo4j_migration.sh
   
   set -e
   
   echo "🚀 L9 Neo4j GraphRAG Deployment"
   echo "================================"
   
   # Validate prerequisites
   echo "1️⃣ Validating prerequisites..."
   docker --version || { echo "❌ Docker not available"; exit 1; }
   docker-compose --version || { echo "❌ Docker Compose not available"; exit 1; }
   
   # Backup existing Kuzu data
   echo "2️⃣ Backing up existing Kuzu data..."
   BACKUP_DIR="./.neural-tools/backups/kuzu-$(date +%Y%m%d-%H%M%S)"
   mkdir -p "$BACKUP_DIR"
   cp -r "./.neural-tools/kuzu" "$BACKUP_DIR/" 2>/dev/null || echo "No Kuzu data to backup"
   
   # Deploy Neo4j container
   echo "3️⃣ Deploying Neo4j container..."
   docker-compose -f docker-compose.neural-tools.yml up -d neo4j-graphrag
   
   # Wait for Neo4j to be ready
   echo "4️⃣ Waiting for Neo4j to be ready..."
   for i in {1..60}; do
       if docker exec ${PROJECT_NAME:-default}-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "RETURN 1" &>/dev/null; then
           echo "✅ Neo4j is ready"
           break
       fi
       echo -n "."
       sleep 2
   done
   
   # Run data migration
   echo "5️⃣ Running data migration..."
   docker exec ${PROJECT_NAME:-default}-neural python3 /app/migrate_kuzu_to_neo4j.py
   
   # Restart Neural Tools server with Neo4j
   echo "6️⃣ Restarting Neural Tools server..."
   docker-compose -f docker-compose.neural-tools.yml restart neural-tools-server
   
   # Run validation tests
   echo "7️⃣ Running validation tests..."
   docker exec ${PROJECT_NAME:-default}-neural python3 -m pytest /app/test_neo4j_integration.py -v
   
   # Update configuration files
   echo "8️⃣ Updating configuration..."
   # Update neural-install.sh with Neo4j support
   # Update documentation
   
   echo "✅ Neo4j migration deployment complete!"
   echo "🔍 Monitor logs: docker logs ${PROJECT_NAME:-default}-neo4j"
   echo "🌐 Neo4j Browser: http://localhost:7474"
   ```

2. **Monitoring Setup**
   ```yaml
   # monitoring/neo4j-monitoring.yml
   version: '3.8'
   
   services:
     neo4j-monitoring:
       image: prom/prometheus:v2.40.0
       container_name: ${PROJECT_NAME}-neo4j-monitoring
       command:
         - '--config.file=/etc/prometheus/prometheus.yml'
         - '--storage.tsdb.path=/prometheus'
         - '--web.console.libraries=/etc/prometheus/console_libraries'
         - '--web.console.templates=/etc/prometheus/consoles'
         - '--storage.tsdb.retention.time=200h'
         - '--web.enable-lifecycle'
       restart: unless-stopped
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml
         - prometheus_data:/prometheus
       ports:
         - "9090:9090"
       networks:
         - neural-network
   
   # prometheus.yml configuration for Neo4j metrics
   global:
     scrape_interval: 15s
   
   scrape_configs:
     - job_name: 'neo4j'
       static_configs:
         - targets: ['neo4j-graphrag:2004']
       metrics_path: /metrics
   ```

3. **Alerting Configuration**
   ```python
   # monitoring/neo4j_health_check.py
   import asyncio
   import logging
   from datetime import datetime
   from neo4j_client import Neo4jGraphClient

   class Neo4jHealthMonitor:
       def __init__(self):
           self.client = Neo4jGraphClient()
           self.logger = logging.getLogger(__name__)
       
       async def check_connectivity(self) -> bool:
           """Check basic Neo4j connectivity"""
           try:
               result = self.client.execute_query("RETURN 1 AS health_check")
               return len(result) == 1 and result[0]['health_check'] == 1
           except Exception as e:
               self.logger.error(f"Neo4j connectivity check failed: {e}")
               return False
       
       async def check_performance(self) -> Dict[str, Any]:
           """Check query performance metrics"""
           try:
               start_time = time.perf_counter()
               result = self.client.execute_query("""
                   MATCH (n)
                   RETURN COUNT(n) AS total_nodes
                   LIMIT 1
               """)
               end_time = time.perf_counter()
               
               query_time = (end_time - start_time) * 1000  # ms
               
               return {
                   'query_time_ms': query_time,
                   'total_nodes': result[0]['total_nodes'],
                   'healthy': query_time < 100  # 100ms threshold
               }
           except Exception as e:
               self.logger.error(f"Neo4j performance check failed: {e}")
               return {'healthy': False, 'error': str(e)}
       
       async def check_data_integrity(self) -> bool:
           """Verify core data relationships exist"""
           try:
               # Check for basic schema integrity
               checks = [
                   "MATCH (f:File) RETURN COUNT(f) > 0 AS has_files",
                   "MATCH (fn:Function) RETURN COUNT(fn) > 0 AS has_functions",
                   "MATCH ()-[r:CALLS]->() RETURN COUNT(r) > 0 AS has_relationships"
               ]
               
               for check_query in checks:
                   result = self.client.execute_query(check_query)
                   if not result[0][list(result[0].keys())[0]]:
                       return False
               
               return True
           except Exception as e:
               self.logger.error(f"Data integrity check failed: {e}")
               return False
       
       async def run_health_check(self) -> Dict[str, Any]:
           """Run complete health check suite"""
           health_status = {
               'timestamp': datetime.now().isoformat(),
               'overall_healthy': True,
               'checks': {}
           }
           
           # Connectivity check
           connectivity = await self.check_connectivity()
           health_status['checks']['connectivity'] = connectivity
           
           if not connectivity:
               health_status['overall_healthy'] = False
               return health_status
           
           # Performance check
           performance = await self.check_performance()
           health_status['checks']['performance'] = performance
           if not performance.get('healthy', False):
               health_status['overall_healthy'] = False
           
           # Data integrity check
           data_integrity = await self.check_data_integrity()
           health_status['checks']['data_integrity'] = data_integrity
           if not data_integrity:
               health_status['overall_healthy'] = False
           
           return health_status

   # Automated monitoring script
   async def monitor_neo4j():
       monitor = Neo4jHealthMonitor()
       
       while True:
           try:
               health_status = await monitor.run_health_check()
               
               if not health_status['overall_healthy']:
                   # Send alert (implement your alerting mechanism)
                   logging.error(f"Neo4j health check failed: {health_status}")
                   # Could integrate with PagerDuty, Slack, etc.
               else:
                   logging.info("Neo4j health check passed")
               
               # Wait 5 minutes between checks
               await asyncio.sleep(300)
               
           except Exception as e:
               logging.error(f"Health monitoring error: {e}")
               await asyncio.sleep(60)  # Shorter retry interval on errors

   if __name__ == "__main__":
       asyncio.run(monitor_neo4j())
   ```

**Exit Criteria for Phase 5:**
- [ ] Neo4j deployed successfully in production environment
- [ ] Health monitoring shows 100% uptime for 24 hours
- [ ] Performance metrics meet SLA requirements  
- [ ] Alerting system tested and functional
- [ ] Rollback procedure documented and tested
- [ ] Operations team trained on Neo4j management

## Dependencies and Prerequisites

### Required Software Versions
```txt
# Core Dependencies
Docker >= 24.0.0
Docker Compose >= 2.20.0
Python >= 3.11
Neo4j Community Edition 5.15.0

# Python Packages
neo4j>=5.15.0,<6.0.0
py2neo>=2021.2.4
neotime>=1.7.4
pytest>=7.0.0
pytest-asyncio>=0.21.0

# Docker Images
neo4j:5.15-community
prom/prometheus:v2.40.0 (optional monitoring)
```

### Environment Variables
```bash
# Neo4j Configuration
NEO4J_AUTH=neo4j/your_secure_password_here
NEO4J_PLUGINS=["apoc", "graph-data-science"]
NEO4J_URI=bolt://neo4j-graphrag:7687
NEO4J_DATABASE=${PROJECT_NAME}_graph

# Port Configuration (optional)
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687

# Migration Configuration
MIGRATION_BATCH_SIZE=1000
MIGRATION_TIMEOUT_SECONDS=300
```

### Infrastructure Requirements
- **CPU**: 2-4 cores for Neo4j container
- **Memory**: 4-8GB RAM (2GB minimum for Neo4j heap)  
- **Storage**: 10GB+ for graph data, logs, and backups
- **Network**: Internal Docker network connectivity

## Risk Assessment and Mitigation

### High-Risk Items

1. **Data Loss During Migration**
   - **Risk Level**: HIGH
   - **Mitigation**: Comprehensive backup strategy, migration validation, rollback procedures
   - **Testing**: Full migration rehearsal in staging environment

2. **Performance Degradation** 
   - **Risk Level**: MEDIUM
   - **Mitigation**: Performance benchmarking, query optimization, connection pooling
   - **Testing**: Load testing with production-equivalent data volumes

3. **Multi-tenancy Isolation Failure**
   - **Risk Level**: MEDIUM  
   - **Mitigation**: Rigorous testing of project isolation, database-per-project design
   - **Testing**: Concurrent multi-project access validation

### Low-Risk Items

1. **Learning Curve for Operations Team**
   - **Risk Level**: LOW
   - **Mitigation**: Comprehensive documentation, training sessions
   
2. **Docker Container Resource Consumption**
   - **Risk Level**: LOW
   - **Mitigation**: Resource monitoring, container limits configuration

## Rollback Strategy

### Immediate Rollback (< 1 hour)
1. **Disable Neo4j**: Set `GRAPHRAG_ENABLED=false` in environment
2. **Restore Kuzu**: Revert to Kuzu client code (git rollback)
3. **Restart Containers**: `docker-compose restart neural-tools-server`
4. **Verify Functionality**: Run basic MCP tool tests

### Complete Rollback (< 4 hours)  
1. **Data Restoration**: Restore Kuzu database from backup
2. **Code Reversion**: Full git revert to pre-migration state
3. **Container Rebuild**: `docker-compose down && docker-compose up --build`
4. **Full Testing**: Execute complete test suite
5. **Stakeholder Notification**: Inform team of rollback completion

### Rollback Success Criteria
- [ ] All MCP tools functioning without Neo4j
- [ ] Kuzu GraphRAG operations working normally
- [ ] Zero data loss confirmed
- [ ] System performance at pre-migration levels
- [ ] Full test suite passing

## Success Metrics and KPIs

### Technical Metrics
- **Reliability**: 0 lock-related failures over 30-day period
- **Performance**: 95th percentile query time < 100ms
- **Scalability**: Support 10+ concurrent projects
- **Uptime**: 99.9% availability for graph operations
- **Data Consistency**: 100% data integrity across restarts

### Operational Metrics  
- **Deployment Time**: Migration completed within 4-day target
- **Rollback Capability**: Rollback tested and executable within 1 hour
- **Team Training**: Operations team 100% trained on Neo4j management
- **Documentation**: Complete operational runbooks delivered

### Business Metrics
- **User Experience**: Zero user-reported issues related to GraphRAG
- **Development Velocity**: No slowdown in feature development
- **Cost Impact**: Infrastructure costs remain within 10% of baseline
- **Risk Reduction**: Elimination of single-writer concurrency bottleneck

## Testing Strategy

### Unit Testing (70% coverage minimum)
```python
# Test Categories
- Neo4j client connectivity and configuration
- Graph query execution and error handling  
- Data migration functions and validation
- MCP tool integration with Neo4j
- Multi-tenancy and project isolation
```

### Integration Testing
```python
# Test Scenarios  
- End-to-end GraphRAG workflow
- Container restart resilience
- Cross-service communication (Neo4j + Qdrant + Nomic)
- Performance under realistic load
- Failure mode handling and recovery
```

### Load Testing
```python
# Performance Scenarios
- Concurrent multi-project access (5+ projects)
- Large dataset operations (10,000+ nodes)
- Complex graph traversal queries
- Peak load simulation (50+ concurrent queries)
- Memory and CPU stress testing
```

### Security Testing
```python
# Security Validation
- Authentication and authorization
- Network isolation verification  
- Data encryption at rest and in transit
- Access control between projects
- Audit logging functionality
```

## Documentation Deliverables

### Technical Documentation
1. **Neo4j Architecture Guide**: Complete system design documentation
2. **Migration Runbook**: Step-by-step migration procedures
3. **API Reference**: Updated MCP tool documentation with Neo4j examples
4. **Troubleshooting Guide**: Common issues and resolution procedures

### Operational Documentation
1. **Deployment Guide**: Production deployment procedures
2. **Monitoring Playbook**: Health monitoring and alerting setup
3. **Backup and Recovery**: Data protection procedures
4. **Performance Tuning**: Optimization guidelines and best practices

## References and Further Reading

### Neo4j Official Documentation
- **Neo4j 5.15 Documentation**: https://neo4j.com/docs/
- **Cypher Query Language**: https://neo4j.com/docs/cypher-manual/5/
- **Neo4j Python Driver**: https://neo4j.com/docs/python-manual/5.15/
- **Docker Deployment**: https://neo4j.com/docs/operations-manual/5/docker/

### GraphRAG and Knowledge Graphs  
- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag
- **Graph-based RAG Patterns**: Research papers and implementations
- **Knowledge Graph Construction**: Best practices for code analysis

### Performance and Monitoring
- **Neo4j Performance Tuning**: https://neo4j.com/docs/operations-manual/5/performance/
- **Prometheus Monitoring**: https://prometheus.io/docs/
- **Graph Database Benchmarking**: Industry benchmarks and comparisons

## Approval and Sign-off

### Technical Approval
- [ ] **Lead Software Engineer**: Technical architecture review and approval
- [ ] **DevOps Engineer**: Infrastructure and deployment review
- [ ] **Quality Assurance**: Testing strategy and criteria approval

### Business Approval  
- [ ] **Engineering Manager**: Resource allocation and timeline approval
- [ ] **Product Owner**: User impact and business value validation
- [ ] **Security Team**: Security and compliance review

### Final Approval
- [ ] **Technical Director**: Final technical sign-off
- [ ] **Project Sponsor**: Business case and investment approval

**Approval Date**: _________________  
**Implementation Start**: _________________  
**Target Completion**: _________________  

---

**Document Status**: Proposed  
**Next Review Date**: 2025-09-06  
**Distribution List**: Engineering Team, DevOps, QA, Management
