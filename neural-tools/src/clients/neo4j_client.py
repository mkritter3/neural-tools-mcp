#!/usr/bin/env python3
"""
Neo4j GraphRAG Client for Neural Tools L9 System
Replaces Kuzu embedded database with Neo4j client-server architecture
Provides graph operations for code relationship mapping and GraphRAG queries

Architecture: Neo4j 5.23.0 Community + Python neo4j driver 5.22.0
Features: Code relationships, semantic search integration, hybrid GraphRAG
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError, ConfigurationError
import json
import hashlib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jGraphRAGClient:
    """
    Neo4j client for GraphRAG operations in L9 Neural Tools system
    Provides isolated, multi-tenant graph database access
    """
    
    def __init__(self, 
                 uri: str = None, 
                 username: str = None, 
                 password: str = None,
                 project_name: str = None):
        """
        Initialize Neo4j client with L9 isolation
        
        Args:
            uri: Neo4j connection URI (default: bolt://neo4j-graph:7687)
            username: Neo4j username (default: neo4j)
            password: Neo4j password (default: neural-l9-2025)
            project_name: Project isolation namespace
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:47687')
        self.username = username or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'graphrag-password')
        self.project_name = project_name or os.getenv('PROJECT_NAME', 'default')
        
        self.driver: Optional[Driver] = None
        self.connection_verified = False
        
        # L9 project isolation - all nodes get project prefix
        self.node_prefix = f"proj_{self._sanitize_project_name(self.project_name)}_"
        
        logger.info(f"Neo4j client initialized for project: {self.project_name}")
    
    def _sanitize_project_name(self, name: str) -> str:
        """Sanitize project name for Neo4j labels"""
        return ''.join(c if c.isalnum() else '_' for c in name.lower())
    
    async def connect(self) -> bool:
        """
        Establish connection to Neo4j database
        
        Returns:
            bool: True if connection successful
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                max_connection_lifetime=30 * 60,  # 30 minutes
                max_connection_pool_size=50,
                connection_acquisition_timeout=60  # 1 minute
            )
            
            # Verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.connection_verified = True
                    logger.info("Neo4j connection verified successfully")
                    
                    # Initialize project schema
                    await self._initialize_project_schema()
                    
                    return True
                    
        except (ServiceUnavailable, AuthError, ConfigurationError) as e:
            logger.error(f"Neo4j connection failed: {str(e)}")
            self.connection_verified = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {str(e)}")
            self.connection_verified = False
            return False
    
    async def _initialize_project_schema(self):
        """Initialize Neo4j schema with constraints and indexes for project"""
        if not self.driver:
            return
            
        constraints_and_indexes = [
            # Unique constraints for project isolation
            f"CREATE CONSTRAINT {self.node_prefix}file_path_unique IF NOT EXISTS FOR (f:File) REQUIRE f.project_path IS UNIQUE",
            f"CREATE CONSTRAINT {self.node_prefix}function_signature_unique IF NOT EXISTS FOR (fn:Function) REQUIRE fn.project_signature IS UNIQUE",
            f"CREATE CONSTRAINT {self.node_prefix}class_name_unique IF NOT EXISTS FOR (c:Class) REQUIRE c.project_name IS UNIQUE",
            
            # Performance indexes
            f"CREATE INDEX {self.node_prefix}file_extension_idx IF NOT EXISTS FOR (f:File) ON (f.extension)",
            f"CREATE INDEX {self.node_prefix}function_name_idx IF NOT EXISTS FOR (fn:Function) ON (fn.name)",
            f"CREATE INDEX {self.node_prefix}class_name_idx IF NOT EXISTS FOR (c:Class) ON (c.name)",
            f"CREATE INDEX {self.node_prefix}embedding_idx IF NOT EXISTS FOR (n) ON (n.embedding_vector)",
            
            # Full-text search indexes for semantic queries
            f"CREATE FULLTEXT INDEX {self.node_prefix}code_content_search IF NOT EXISTS FOR (n:File|Function|Class) ON EACH [n.content, n.docstring, n.comments]",
            f"CREATE FULLTEXT INDEX {self.node_prefix}semantic_search IF NOT EXISTS FOR (n) ON EACH [n.semantic_summary, n.description]"
        ]
        
        try:
            with self.driver.session() as session:
                for query in constraints_and_indexes:
                    try:
                        session.run(query)
                        logger.debug(f"Applied schema: {query[:50]}...")
                    except Exception as e:
                        # Ignore if constraint/index already exists
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Schema application warning: {str(e)[:100]}")
                            
            logger.info(f"Neo4j schema initialized for project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {str(e)}")
    
    async def close(self):
        """Close Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            self.connection_verified = False
            logger.info("Neo4j connection closed")
    
    def _get_project_labels(self, base_label: str) -> str:
        """Get project-isolated label"""
        return f"{self.node_prefix}{base_label}"
    
    def _add_project_metadata(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Add project isolation metadata to node properties"""
        properties["project"] = self.project_name
        properties["created_at"] = datetime.utcnow().isoformat()
        return properties
    
    async def create_file_node(self, 
                             file_path: str, 
                             content: str, 
                             extension: str,
                             size_bytes: int,
                             embedding_vector: Optional[List[float]] = None) -> bool:
        """
        Create or update a file node in the graph
        
        Args:
            file_path: Relative path from project root
            content: File content for semantic analysis
            extension: File extension (.py, .js, etc.)
            size_bytes: File size
            embedding_vector: Optional semantic embedding
            
        Returns:
            bool: True if successful
        """
        if not self.connection_verified:
            logger.error("Neo4j not connected")
            return False
        
        try:
            # Create project-unique identifier
            project_path = f"{self.project_name}::{file_path}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            properties = self._add_project_metadata({
                "path": file_path,
                "project_path": project_path,
                "content": content[:10000],  # Limit content size
                "extension": extension,
                "size_bytes": size_bytes,
                "content_hash": content_hash,
                "lines_count": content.count('\n') + 1,
                "embedding_vector": embedding_vector
            })
            
            query = f"""
            MERGE (f:File {{project_path: $project_path}})
            SET f += $properties
            SET f.updated_at = datetime()
            RETURN f.project_path as path
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "project_path": project_path,
                    "properties": properties
                })
                
                created_path = result.single()["path"]
                logger.debug(f"File node created/updated: {created_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating file node: {str(e)}")
            return False
    
    async def create_function_node(self, 
                                 file_path: str,
                                 function_name: str, 
                                 signature: str,
                                 docstring: str = "",
                                 start_line: int = 0,
                                 end_line: int = 0,
                                 embedding_vector: Optional[List[float]] = None) -> bool:
        """
        Create a function node and link to its file
        
        Args:
            file_path: Parent file path
            function_name: Function name
            signature: Full function signature
            docstring: Function documentation
            start_line: Starting line number
            end_line: Ending line number
            embedding_vector: Optional semantic embedding
            
        Returns:
            bool: True if successful
        """
        if not self.connection_verified:
            return False
        
        try:
            project_signature = f"{self.project_name}::{file_path}::{function_name}::{signature}"
            project_file_path = f"{self.project_name}::{file_path}"
            
            properties = self._add_project_metadata({
                "name": function_name,
                "signature": signature,
                "project_signature": project_signature,
                "docstring": docstring,
                "start_line": start_line,
                "end_line": end_line,
                "line_count": end_line - start_line + 1,
                "embedding_vector": embedding_vector
            })
            
            query = f"""
            MERGE (fn:Function {{project_signature: $project_signature}})
            SET fn += $properties
            SET fn.updated_at = datetime()
            WITH fn
            MATCH (f:File {{project_path: $project_file_path}})
            MERGE (f)-[:CONTAINS]->(fn)
            RETURN fn.name as name
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "project_signature": project_signature,
                    "project_file_path": project_file_path,
                    "properties": properties
                })
                
                created_name = result.single()["name"]
                logger.debug(f"Function node created: {created_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating function node: {str(e)}")
            return False
    
    async def create_class_node(self, 
                              file_path: str,
                              class_name: str, 
                              docstring: str = "",
                              start_line: int = 0,
                              end_line: int = 0,
                              embedding_vector: Optional[List[float]] = None) -> bool:
        """
        Create a class node and link to its file
        
        Args:
            file_path: Parent file path
            class_name: Class name
            docstring: Class documentation
            start_line: Starting line number
            end_line: Ending line number
            embedding_vector: Optional semantic embedding
            
        Returns:
            bool: True if successful
        """
        if not self.connection_verified:
            return False
        
        try:
            project_class_name = f"{self.project_name}::{file_path}::{class_name}"
            project_file_path = f"{self.project_name}::{file_path}"
            
            properties = self._add_project_metadata({
                "name": class_name,
                "project_name": project_class_name,
                "docstring": docstring,
                "start_line": start_line,
                "end_line": end_line,
                "line_count": end_line - start_line + 1,
                "embedding_vector": embedding_vector
            })
            
            query = f"""
            MERGE (c:Class {{project_name: $project_class_name}})
            SET c += $properties
            SET c.updated_at = datetime()
            WITH c
            MATCH (f:File {{project_path: $project_file_path}})
            MERGE (f)-[:CONTAINS]->(c)
            RETURN c.name as name
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "project_class_name": project_class_name,
                    "project_file_path": project_file_path,
                    "properties": properties
                })
                
                created_name = result.single()["name"]
                logger.debug(f"Class node created: {created_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating class node: {str(e)}")
            return False
    
    async def create_dependency_relationship(self, 
                                           source_file: str, 
                                           target_file: str, 
                                           dependency_type: str = "IMPORTS") -> bool:
        """
        Create dependency relationship between files
        
        Args:
            source_file: Source file path
            target_file: Target file path
            dependency_type: Type of dependency (IMPORTS, CALLS, EXTENDS, etc.)
            
        Returns:
            bool: True if successful
        """
        if not self.connection_verified:
            return False
        
        try:
            source_path = f"{self.project_name}::{source_file}"
            target_path = f"{self.project_name}::{target_file}"
            
            query = f"""
            MATCH (source:File {{project_path: $source_path}})
            MATCH (target:File {{project_path: $target_path}})
            MERGE (source)-[r:{dependency_type}]->(target)
            SET r.created_at = datetime()
            SET r.project = $project
            RETURN type(r) as relationship
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "source_path": source_path,
                    "target_path": target_path,
                    "project": self.project_name
                })
                
                relationship = result.single()["relationship"]
                logger.debug(f"Dependency created: {source_file} -{relationship}-> {target_file}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating dependency: {str(e)}")
            return False
    
    async def semantic_graph_search(self, 
                                  query_text: str, 
                                  limit: int = 10,
                                  node_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search across the code graph
        
        Args:
            query_text: Natural language query
            limit: Maximum results to return
            node_types: Filter by node types (File, Function, Class)
            
        Returns:
            List of matching nodes with relevance scores
        """
        if not self.connection_verified:
            return []
        
        try:
            # Default to all node types
            if not node_types:
                node_types = ["File", "Function", "Class"]
            
            # Use Neo4j's full-text search
            search_labels = "|".join(node_types)
            
            query = f"""
            CALL db.index.fulltext.queryNodes(
                '{self.node_prefix}code_content_search', 
                $query_text
            ) YIELD node, score
            WHERE node.project = $project
            AND any(label IN labels(node) WHERE label IN $node_types)
            RETURN 
                node.name as name,
                node.path as path,
                node.project_signature as signature,
                labels(node) as types,
                node.docstring as docstring,
                score,
                node.start_line as start_line,
                node.end_line as end_line
            ORDER BY score DESC
            LIMIT $limit
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "query_text": query_text,
                    "project": self.project_name,
                    "node_types": node_types,
                    "limit": limit
                })
                
                results = []
                for record in result:
                    results.append({
                        "name": record["name"],
                        "path": record["path"],
                        "signature": record["signature"],
                        "types": record["types"],
                        "docstring": record["docstring"],
                        "relevance_score": float(record["score"]),
                        "start_line": record["start_line"],
                        "end_line": record["end_line"]
                    })
                
                logger.info(f"Semantic search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    async def get_code_dependencies(self, 
                                  file_path: str, 
                                  max_depth: int = 3) -> Dict[str, Any]:
        """
        Get dependency graph for a specific file
        
        Args:
            file_path: Target file path
            max_depth: Maximum traversal depth
            
        Returns:
            Dependency graph structure
        """
        if not self.connection_verified:
            return {}
        
        try:
            project_file_path = f"{self.project_name}::{file_path}"
            
            query = f"""
            MATCH path = (f:File {{project_path: $project_file_path}})-[*1..{max_depth}]-(connected)
            WHERE connected.project = $project
            RETURN 
                f.path as source_file,
                collect(DISTINCT {{
                    path: connected.path,
                    name: connected.name,
                    type: head(labels(connected)),
                    relationship: [r in relationships(path) | type(r)]
                }}) as dependencies
            """
            
            with self.driver.session() as session:
                result = session.run(query, {
                    "project_file_path": project_file_path,
                    "project": self.project_name
                })
                
                record = result.single()
                if record:
                    return {
                        "source_file": record["source_file"],
                        "dependencies": record["dependencies"],
                        "dependency_count": len(record["dependencies"])
                    }
                else:
                    return {"source_file": file_path, "dependencies": [], "dependency_count": 0}
                
        except Exception as e:
            logger.error(f"Error getting dependencies: {str(e)}")
            return {}
    
    async def execute_cypher_query(self, 
                                 query: str, 
                                 parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute custom Cypher query with project isolation
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if not self.connection_verified:
            return []
        
        try:
            # Add project isolation to parameters
            if parameters is None:
                parameters = {}
            parameters["project"] = self.project_name
            parameters["node_prefix"] = self.node_prefix
            
            with self.driver.session() as session:
                result = session.run(query, parameters)
                
                records = []
                for record in result:
                    records.append(dict(record))
                
                logger.debug(f"Cypher query returned {len(records)} records")
                return records
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {str(e)}")
            return []
    
    async def get_project_statistics(self) -> Dict[str, Any]:
        """
        Get project graph statistics
        
        Returns:
            Project statistics dictionary
        """
        if not self.connection_verified:
            return {}
        
        try:
            query = """
            MATCH (n) WHERE n.project = $project
            WITH labels(n)[0] as node_type, count(n) as count
            RETURN collect({type: node_type, count: count}) as node_stats
            """
            
            with self.driver.session() as session:
                result = session.run(query, {"project": self.project_name})
                record = result.single()
                
                stats = {
                    "project": self.project_name,
                    "node_statistics": record["node_stats"] if record else [],
                    "connection_status": "connected" if self.connection_verified else "disconnected"
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting project statistics: {str(e)}")
            return {"project": self.project_name, "error": str(e)}
    
    async def clear_project_data(self) -> bool:
        """
        Clear all project data (for testing/migration)
        WARNING: This deletes all nodes and relationships for the project
        
        Returns:
            bool: True if successful
        """
        if not self.connection_verified:
            return False
        
        try:
            query = """
            MATCH (n) WHERE n.project = $project
            DETACH DELETE n
            RETURN count(*) as deleted_count
            """
            
            with self.driver.session() as session:
                result = session.run(query, {"project": self.project_name})
                deleted_count = result.single()["deleted_count"]
                
                logger.warning(f"Cleared {deleted_count} nodes for project: {self.project_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing project data: {str(e)}")
            return False

# Async context manager support
class AsyncNeo4jClient:
    """Async context manager wrapper for Neo4jGraphRAGClient"""
    
    def __init__(self, **kwargs):
        self.client = Neo4jGraphRAGClient(**kwargs)
    
    async def __aenter__(self):
        await self.client.connect()
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

# Factory function for easy client creation
def create_neo4j_client(project_name: str = None) -> Neo4jGraphRAGClient:
    """
    Factory function to create Neo4j client with environment defaults
    
    Args:
        project_name: Project name for isolation
        
    Returns:
        Configured Neo4jGraphRAGClient instance
    """
    return Neo4jGraphRAGClient(project_name=project_name)

# Health check utility
async def neo4j_health_check() -> Dict[str, Any]:
    """
    Perform Neo4j health check
    
    Returns:
        Health check results
    """
    try:
        async with AsyncNeo4jClient() as client:
            stats = await client.get_project_statistics()
            return {
                "status": "healthy",
                "connection": "successful",
                "project_stats": stats
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    # Simple test
    async def test_client():
        async with AsyncNeo4jClient(project_name="test-project") as client:
            # Test basic operations
            await client.create_file_node(
                file_path="test.py",
                content="def hello(): pass",
                extension=".py",
                size_bytes=100
            )
            
            stats = await client.get_project_statistics()
            print(f"Project stats: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_client())