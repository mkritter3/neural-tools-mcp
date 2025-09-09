"""
Core MCP Tools Module
Provides essential project understanding and search functionality using service container pattern.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP

from services.service_container import ServiceContainer

logger = logging.getLogger(__name__)

def register_core_tools(mcp: FastMCP, container=None):
    """Register all core MCP tools with global variable pattern (FastMCP-compatible)"""
    
    @mcp.tool()
    async def project_understanding(
        scope: str = "full",
        max_tokens: str = "2000"
    ) -> Dict[str, Any]:
        """Generate condensed project understanding without reading all files
        
        Args:
            scope: 'full', 'architecture', 'dependencies', 'core_logic'
            max_tokens: Maximum tokens for response (string converted to int)
        """
        try:
            from neural_server_refactored import ensure_services_initialized, qdrant_client, PROJECT_NAME
            await ensure_services_initialized()
            
            # Convert string parameters to proper types
            max_tokens_int = int(max_tokens) if isinstance(max_tokens, str) else max_tokens
            
            understanding = {
                "project": PROJECT_NAME,
                "timestamp": datetime.now().isoformat(),
                "scope": scope,
                "max_tokens": max_tokens_int
            }
            
            # Get high-level stats from Qdrant collections
            if not qdrant_client:
                return {
                    "status": "error",
                    "message": "Qdrant client not initialized"
                }
            
            try:
                collection_names = await qdrant_client.get_collections()
                collection_prefix = f"project_{PROJECT_NAME}_"
                project_collections = [name for name in collection_names if name.startswith(collection_prefix)]
                
                understanding["indexed_categories"] = [
                    c.replace(collection_prefix, "") for c in project_collections
                ]
                
                # Sample semantic clusters from main collection
                code_collection = f"{collection_prefix}code"
                if code_collection in project_collections:
                    # Get diverse samples using MMR-style selection
                    # Determine embed dimension (default 768)
                    import os
                    try:
                        embed_dim = int(os.environ.get('EMBED_DIM', os.environ.get('EMBEDDING_DIMENSION', '768')))
                    except Exception:
                        embed_dim = 768
                    neutral = [0.1] * embed_dim
                    search_results = await qdrant_client.search_vectors(
                        collection_name=code_collection,
                        query_vector=neutral,
                        limit=10
                    )
                    
                    patterns = []
                    for hit in search_results:
                        if isinstance(hit, dict):
                            payload = hit.get('payload', {})
                        else:
                            payload = getattr(hit, 'payload', {}) or {}
                        patterns.append({
                            "type": payload.get("type", "unknown"),
                            "category": payload.get("category", "general"),
                            "summary": (payload.get("content", "") or "")[:100]
                        })
                    understanding["code_patterns"] = patterns
            except Exception as e:
                logger.warning(f"Qdrant collection analysis failed: {e}")
            
            # Get graph structure from Neo4j if available
            from neural_server_refactored import neo4j_client
            if neo4j_client and neo4j_client.initialized:
                try:
                    # Get file and function counts
                    arch_result = await neo4j_client.execute_cypher("""
                        MATCH (f:File)
                        OPTIONAL MATCH (f)-[:CONTAINS]->(fn:Function)
                        OPTIONAL MATCH (f)-[:CONTAINS]->(c:Class)
                        RETURN 
                            COUNT(DISTINCT f) as file_count,
                            COUNT(DISTINCT fn) as function_count,
                            COUNT(DISTINCT c) as class_count
                    """)
                    
                    if arch_result.get("status") == "success" and arch_result.get("records"):
                        record = arch_result["records"][0]
                        understanding["architecture"] = {
                            "structure": [
                                {"type": "files", "count": record.get('file_count', 0)},
                                {"type": "functions", "count": record.get('function_count', 0)},
                                {"type": "classes", "count": record.get('class_count', 0)}
                            ],
                            "database": "neo4j"
                        }
                        
                        # Get core relationships
                        rel_result = await neo4j_client.execute_cypher("""
                            MATCH ()-[r:CALLS]->()
                            RETURN 'CALLS' as rel_type, COUNT(r) as count
                            UNION
                            MATCH ()-[r:CONTAINS]->()
                            RETURN 'CONTAINS' as rel_type, COUNT(r) as count
                            ORDER BY count DESC
                        """)
                        
                        if rel_result.get("status") == "success" and rel_result.get("records"):
                            understanding["relationships"] = [
                                {"type": row.get("rel_type"), "count": row.get("count")} 
                                for row in rel_result["records"]
                            ]
                except Exception as e:
                    logger.warning(f"Neo4j graph structure query failed: {e}")
            
            # Compress to fit token limit
            response_str = json.dumps(understanding, indent=2)
            if len(response_str) // 4 > max_tokens_int:
                # Truncate less important sections
                if "code_patterns" in understanding:
                    understanding["code_patterns"] = understanding["code_patterns"][:3]
                response_str = json.dumps(understanding, indent=2)
            
            return {
                "status": "success",
                "understanding": understanding,
                "token_count": len(response_str) // 4
            }
            
        except Exception as e:
            logger.error(f"Project understanding failed: {e}")
            return {"status": "error", "message": str(e)}
    
    @mcp.tool()
    async def semantic_code_search(
        query: str,
        search_type: str = "semantic",
        limit: str = "10",
        min_score: str = "0.7",
        use_prism: bool = True,
        prism_boost: str = "0.3"
    ) -> Dict[str, Any]:
        """Search code by meaning, not just text matching
        
        Args:
            query: Natural language query (e.g., 'error handling logic')
            search_type: 'semantic', 'hybrid', 'exact'
            limit: Maximum results
            min_score: Minimum similarity score
            use_prism: Whether to boost results with PRISM importance scores
            prism_boost: How much to blend PRISM scores (0.0-1.0)
        """
        try:
            # Convert string parameters to proper types
            limit_int = int(limit) if isinstance(limit, str) else limit
            min_score_float = float(min_score) if isinstance(min_score, str) else min_score
            prism_boost_float = float(prism_boost) if isinstance(prism_boost, str) else prism_boost
            
            # Get services
            from neural_server_refactored import ensure_services_initialized, nomic_client, qdrant_client, PROJECT_NAME
            await ensure_services_initialized()
            
            if not nomic_client:
                return {"status": "error", "message": "Nomic service not initialized", "results": []}
            if not qdrant_client:
                return {"status": "error", "message": "Qdrant service not initialized", "results": []}
            
            # Generate embedding for semantic query
            embeddings = await nomic_client.get_embeddings([query])
            query_vector = embeddings[0]
            
            collection_name = f"project_{PROJECT_NAME}_code"
            
            if search_type == "semantic":
                # Pure vector search
                search_results = await qdrant_client.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit_int,
                    score_threshold=min_score_float
                )
            elif search_type == "hybrid":
                # Use RRF hybrid search from QdrantService
                search_results = await qdrant_client.rrf_hybrid_search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_text=query,
                    limit=limit_int
                )
            else:  # exact - text search using scroll
                search_results = await qdrant_client.scroll_collection(
                    collection_name=collection_name,
                    limit=limit_int,
                    text_filter=query
                )
            
            # Format results with token efficiency
            formatted_results = []
            total_tokens = 0
            
            for hit in search_results:
                content = hit.get("content", "")
                # Extract relevant snippet around matches
                snippet_size = 200
                snippet = content[:snippet_size] + "..." if len(content) > snippet_size else content
                
                formatted_results.append({
                    "score": hit.get("score", 1.0),
                    "file_path": hit.get("file_path", "unknown"),
                    "type": hit.get("type", "code"),
                    "snippet": snippet,
                    "full_content_id": hit.get("id")
                })
                total_tokens += len(snippet) // 4
            
            # Note: PRISM boosting not implemented in refactored version yet
            # This would require extracting the PRISM scorer as well
            
            return {
                "status": "success",
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "token_usage": total_tokens,
                "prism_enabled": False  # TODO: Implement PRISM integration
            }
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {"status": "error", "message": str(e)}
    
    @mcp.tool()
    async def atomic_dependency_tracer(
        target: str,
        trace_type: str = "calls",
        max_depth: str = "5",
        include_imports: bool = True
    ) -> Dict[str, Any]:
        """Trace atomic dependencies for functions, classes, or modules
        
        Args:
            target: Function/class/module name to trace
            trace_type: 'calls' (who calls this), 'dependencies' (what this calls), 'both'
            max_depth: Maximum traversal depth
            include_imports: Include import dependencies
        """
        try:
            # Convert string parameters to proper types
            max_depth_int = int(max_depth) if isinstance(max_depth, str) else max_depth
            
            from neural_server_refactored import ensure_services_initialized, neo4j_client
            await ensure_services_initialized()
            
            if not neo4j_client:
                return {"status": "error", "message": "Neo4j GraphRAG not available"}
            
            results = {
                "target": target,
                "trace_type": trace_type,
                "dependencies": []
            }
            
            # Build Neo4j query based on trace type
            if trace_type == "calls":
                # Find functions that call the target
                cypher_query = """
                MATCH (caller:Function)-[:CALLS]->(target:Function)
                WHERE target.name = $target
                RETURN caller.name as caller_name, caller.file_path as caller_file, 
                       target.name as target_name, target.file_path as target_file
                LIMIT 50
                """
            elif trace_type == "dependencies":
                # Find what the target function calls
                cypher_query = """
                MATCH (source:Function)-[:CALLS]->(dependency:Function)
                WHERE source.name = $target
                RETURN source.name as source_name, source.file_path as source_file,
                       dependency.name as dependency_name, dependency.file_path as dependency_file
                LIMIT 50
                """
            else:  # both
                cypher_query = """
                MATCH (f1:Function)-[:CALLS]-(f2:Function)
                WHERE f1.name = $target OR f2.name = $target
                RETURN f1.name as func1_name, f1.file_path as func1_file,
                       f2.name as func2_name, f2.file_path as func2_file
                LIMIT 100
                """
            
            result = await neo4j_client.execute_cypher(cypher_query, {"target": target})
            
            if result.get("status") == "success" and result.get("records"):
                for row in result["records"]:
                    if trace_type == "calls" and row.get('caller_name'):
                        results["dependencies"].append({
                            "type": "function_call",
                            "caller": row['caller_name'],
                            "caller_file": row['caller_file'],
                            "target": row['target_name'],
                            "target_file": row['target_file']
                        })
                    elif trace_type == "dependencies" and row.get('dependency_name'):
                        results["dependencies"].append({
                            "type": "function_dependency",
                            "source": row['source_name'],
                            "source_file": row['source_file'],
                            "dependency": row['dependency_name'],
                            "dependency_file": row['dependency_file']
                        })
                    elif trace_type == "both":
                        results["dependencies"].append({
                            "type": "function_relationship",
                            "func1": row['func1_name'],
                            "func1_file": row['func1_file'],
                            "func2": row['func2_name'],
                            "func2_file": row['func2_file']
                        })
            
            # Add token usage estimate
            results["token_estimate"] = len(str(results)) // 4
            
            return {
                "status": "success",
                "results": results,
                "paths_found": len(results["dependencies"])
            }
            
        except Exception as e:
            logger.error(f"Dependency trace failed: {e}")
            return {"status": "error", "message": str(e)}
    
    @mcp.tool()
    async def vibe_preservation(
        action: str,
        code_sample: Optional[str] = None,
        target_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Preserve and apply project coding style/patterns
        
        Args:
            action: 'learn' (from sample), 'apply' (to target), 'show' (current patterns)
            code_sample: Code to learn style from
            target_file: File to apply style to
        """
        try:
            from neural_server_refactored import ensure_services_initialized, qdrant_client, nomic_client, PROJECT_NAME
            await ensure_services_initialized()
            
            if not qdrant_client:
                return {"status": "error", "message": "Qdrant service not initialized"}
            
            collection_name = f"project_{PROJECT_NAME}_style_patterns"
            
            if action == "learn":
                if not code_sample:
                    return {"status": "error", "message": "code_sample required for learning"}
                
                if not nomic_client or not nomic_client.initialized:
                    return {"status": "error", "message": "Nomic service not initialized"}
                
                # Extract basic style patterns
                patterns = {
                    "indent_style": "spaces" if "    " in code_sample else "tabs",
                    "quote_style": "double" if '"' in code_sample else "single",
                    "semicolons": "yes" if ";" in code_sample else "no",
                    "bracket_style": "same_line" if "{\n" not in code_sample else "new_line",
                    "naming_convention": "camelCase" if any(c.isupper() for c in code_sample) else "snake_case"
                }
                
                # Generate embedding for the style
                embeddings = await nomic_client.get_embeddings([code_sample])
                style_vector = embeddings[0]
                
                # Ensure collection exists
                await qdrant_client.ensure_collection(collection_name)
                
                # Store in Qdrant with named vectors
                from qdrant_client.models import PointStruct
                point = PointStruct(
                    id=hash(code_sample) % 1000000,
                    vector={"dense": style_vector},
                    payload={
                        "patterns": patterns,
                        "sample": code_sample[:500],
                        "learned_at": datetime.now().isoformat()
                    }
                )
                
                result = await qdrant_client.upsert_points(collection_name, [point])
                
                return {
                    "status": "success",
                    "learned_patterns": patterns,
                    "stored": result.get("status") == "success"
                }
                
            elif action == "show":
                # Retrieve stored style patterns (simplified)
                return {
                    "status": "success",
                    "common_patterns": {"style": "analysis not fully implemented in refactored version"},
                    "samples_analyzed": 0
                }
                
            elif action == "apply":
                return {
                    "status": "success",
                    "message": "Style application would transform code to match learned patterns"
                }
            
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Vibe preservation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    @mcp.tool()
    async def project_auto_index(
        scope: str = "modified",
        since_minutes: Optional[str] = None,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Smart auto-indexing tool that indexes project files on-demand
        
        Args:
            scope: Index scope - 'modified' (changed files), 'all' (full reindex), 'git-changes' (uncommitted)
            since_minutes: Only index files modified in last N minutes
            file_patterns: Optional file patterns to index (e.g., ['*.py', '*.js'])
        """
        try:
            # Simplified implementation for refactored version
            from neural_server_refactored import ensure_services_initialized, qdrant_client, nomic_client, PROJECT_NAME
            await ensure_services_initialized()
            
            if not qdrant_client:
                return {"status": "error", "message": "Qdrant service not initialized"}
            if not nomic_client:
                return {"status": "error", "message": "Nomic service not initialized"}
                
            # TODO: Implement full file indexing logic
            # For now, return a status indicating the service is available
            return {
                "status": "success",
                "scope": scope,
                "files_checked": 0,
                "files_indexed": 0,
                "message": "Auto-indexing service available (full implementation pending in refactored version)"
            }
            
        except Exception as e:
            logger.error(f"Auto-indexing failed: {e}")
            return {"status": "error", "message": str(e)}
    
    @mcp.tool()
    async def graph_query(query: str) -> Dict[str, Any]:
        """Execute Cypher query on Neo4j graph database
        
        Args:
            query: Cypher query to execute
        """
        try:
            from neural_server_refactored import ensure_services_initialized, neo4j_client, PROJECT_NAME
            await ensure_services_initialized()
            
            if not neo4j_client:
                return {"status": "error", "message": "Neo4j GraphRAG not available"}
            
            # Execute query on Neo4j
            result = await neo4j_client.execute_cypher(query)
            
            return {
                "status": "success" if result.get("status") == "success" else "error",
                "results": result.get("records", []),
                "count": len(result.get("records", [])),
                "database": "neo4j",
                "project": PROJECT_NAME,
                "message": result.get("message", "")
            }
            
        except Exception as e:
            logger.error(f"Neo4j graph query failed: {e}")
            return {"status": "error", "message": str(e), "query": query}
