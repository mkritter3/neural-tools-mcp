#!/usr/bin/env python3
"""
L9 Neural MCP Server V2 - Enhanced with Tree-sitter Multi-language AST
Production Docker version with deep code understanding
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import asdict

# Add neural-system to path
sys.path.append('/app')

# MCP SDK
from mcp.server.fastmcp import FastMCP
import mcp.types as types

# Qdrant client for hybrid search
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from fastembed import TextEmbedding, SparseTextEmbedding

# Tree-sitter AST analyzer
from tree_sitter_ast import TreeSitterAnalyzer, CodeStructure, TREE_SITTER_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastMCP
mcp = FastMCP("l9-neural-v2")

# Global clients
qdrant_client = None
dense_model = None
sparse_model = None
ast_analyzer = None

# Constants
QDRANT_HOST = os.environ.get('QDRANT_HOST', 'qdrant')
QDRANT_GRPC_PORT = int(os.environ.get('QDRANT_GRPC_PORT', 6334))
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'default')
COLLECTION_PREFIX = f"project_{PROJECT_NAME}_"

async def initialize():
    """Initialize Qdrant, embedding models, and tree-sitter analyzer"""
    global qdrant_client, dense_model, sparse_model, ast_analyzer
    
    try:
        # Connect to Qdrant using gRPC (3-4x faster)
        qdrant_client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_GRPC_PORT,
            prefer_grpc=True
        )
        
        # Initialize embedding models
        dense_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
        sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Initialize tree-sitter analyzer
        if TREE_SITTER_AVAILABLE:
            ast_analyzer = TreeSitterAnalyzer()
            logger.info("✅ Tree-sitter analyzer initialized with multi-language support")
        else:
            logger.warning("⚠️  Tree-sitter not available - code analysis will be limited")
        
        # Ensure default collections exist
        await ensure_collection(f"{COLLECTION_PREFIX}memory")
        await ensure_collection(f"{COLLECTION_PREFIX}code")
        
        logger.info(f"🚀 L9 Neural MCP Server V2 initialized - Project: {PROJECT_NAME}")
        logger.info(f"📍 Qdrant: {QDRANT_HOST}:{QDRANT_GRPC_PORT}")
        logger.info(f"🌳 Tree-sitter: {'Available' if ast_analyzer else 'Unavailable'}")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise

async def ensure_collection(collection_name: str):
    """Ensure collection exists with hybrid search config"""
    try:
        collections = qdrant_client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            # Create with both dense and sparse vectors + payload indexing
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(size=768, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=models.Modifier.IDF)
                }
            )
            
            # Create payload indexes for fast filtering
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="language",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="file_path",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=collection_name,
                field_name="ast_patterns",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Created collection with indexes: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to ensure collection: {e}")

@mcp.tool()
async def memory_store(
    content: str,
    category: str = "general",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Store content in neural memory with hybrid indexing
    
    Args:
        content: Text content to store
        category: Category for organization (general, code, docs, etc.)
        metadata: Additional metadata to store
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}{category}"
        await ensure_collection(collection_name)
        
        # Generate both dense and sparse embeddings
        dense_embedding = list(dense_model.embed([content]))[0]
        sparse_embedding = list(sparse_model.embed([content]))[0]
        
        # Convert sparse embedding to Qdrant format
        sparse_vector = models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )
        
        # Store in Qdrant
        point_id = hash(content + str(datetime.now()))
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=abs(point_id) % (10 ** 8),
                    vector={"dense": dense_embedding.tolist()},
                    sparse_vector={"sparse": sparse_vector},
                    payload={
                        "content": content,
                        "category": category,
                        "timestamp": datetime.now().isoformat(),
                        **(metadata or {})
                    }
                )
            ]
        )
        
        return {
            "status": "success",
            "id": abs(point_id) % (10 ** 8),
            "collection": collection_name
        }
        
    except Exception as e:
        logger.error(f"Store failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def memory_search(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    mode: str = "hybrid"
) -> List[Dict[str, Any]]:
    """Search neural memory using semantic, keyword, or hybrid search
    
    Args:
        query: Search query
        category: Optional category filter
        limit: Number of results
        mode: Search mode (semantic, keyword, hybrid)
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}{category or 'memory'}"
        
        # Generate embeddings based on mode
        results = []
        
        if mode in ["semantic", "hybrid"]:
            # Dense vector search
            dense_embedding = list(dense_model.embed([query]))[0]
            semantic_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=("dense", dense_embedding.tolist()),
                limit=limit
            )
            results.extend([
                {
                    "content": hit.payload.get("content"),
                    "score": hit.score,
                    "type": "semantic",
                    **hit.payload
                }
                for hit in semantic_results
            ])
        
        if mode in ["keyword", "hybrid"]:
            # Sparse vector search (BM25-like)
            sparse_embedding = list(sparse_model.embed([query]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
            
            keyword_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=models.NamedSparseVector(
                    name="sparse",
                    vector=sparse_vector
                ),
                limit=limit
            )
            
            # Merge results if hybrid
            if mode == "hybrid":
                # Deduplicate and combine scores
                seen_content = {r["content"] for r in results}
                for hit in keyword_results:
                    content = hit.payload.get("content")
                    if content not in seen_content:
                        results.append({
                            "content": content,
                            "score": hit.score,
                            "type": "keyword",
                            **hit.payload
                        })
            else:
                results = [
                    {
                        "content": hit.payload.get("content"),
                        "score": hit.score,
                        "type": "keyword",
                        **hit.payload
                    }
                    for hit in keyword_results
                ]
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

@mcp.tool()
async def code_analyze_file(
    file_path: str,
    store_analysis: bool = True
) -> Dict[str, Any]:
    """Analyze a single code file with tree-sitter AST
    
    Args:
        file_path: Path to code file (relative to /app/project)
        store_analysis: Whether to store analysis in memory for search
    """
    try:
        if not ast_analyzer:
            return {"status": "error", "message": "Tree-sitter analyzer not available"}
        
        # Full path within container
        full_path = Path("/app/project") / file_path
        
        if not full_path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}
        
        # Analyze with tree-sitter
        structure = ast_analyzer.analyze_file(str(full_path))
        
        if not structure:
            return {"status": "error", "message": f"Failed to analyze {file_path}"}
        
        # Convert to serializable format
        analysis = {
            "file_path": file_path,
            "language": structure.language.value,
            "functions": [asdict(f) for f in structure.functions],
            "classes": [asdict(c) for c in structure.classes],
            "imports": [asdict(i) for i in structure.imports],
            "complexity_score": structure.complexity_score,
            "dependencies": structure.dependencies,
            "timestamp": datetime.now().isoformat()
        }
        
        # Optionally store in memory for search
        if store_analysis:
            # Generate searchable patterns
            search_patterns = ast_analyzer.extract_searchable_patterns(structure)
            
            # Store code structure for search
            await memory_store(
                content=f"File: {file_path}\nLanguage: {structure.language.value}\nFunctions: {[f.name for f in structure.functions]}\nClasses: {[c.name for c in structure.classes]}\nComplexity: {structure.complexity_score}",
                category="code",
                metadata={
                    "file_path": file_path,
                    "language": structure.language.value,
                    "ast_patterns": search_patterns,
                    "complexity_score": structure.complexity_score,
                    "function_count": len(structure.functions),
                    "class_count": len(structure.classes),
                    "analysis": analysis
                }
            )
        
        return {
            "status": "success",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def code_index_directory(
    directory_path: str = ".",
    extensions: List[str] = [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"],
    max_files: int = 1000
) -> Dict[str, Any]:
    """Index all code files in a directory with tree-sitter analysis
    
    Args:
        directory_path: Directory to index (relative to /app/project)
        extensions: File extensions to include
        max_files: Maximum number of files to process
    """
    try:
        if not ast_analyzer:
            return {"status": "error", "message": "Tree-sitter analyzer not available"}
        
        # Full path within container
        full_path = Path("/app/project") / directory_path
        
        if not full_path.exists():
            return {"status": "error", "message": f"Directory not found: {directory_path}"}
        
        # Find all code files
        code_files = []
        for ext in extensions:
            code_files.extend(full_path.rglob(f"*{ext}"))
        
        code_files = code_files[:max_files]  # Limit processing
        
        indexed_count = 0
        skipped_count = 0
        languages_found = set()
        
        for file_path in code_files:
            try:
                # Get relative path
                rel_path = file_path.relative_to(Path("/app/project"))
                
                # Analyze file
                result = await code_analyze_file(str(rel_path), store_analysis=True)
                
                if result["status"] == "success":
                    indexed_count += 1
                    languages_found.add(result["analysis"]["language"])
                else:
                    skipped_count += 1
                    logger.warning(f"Skipped {rel_path}: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                skipped_count += 1
                logger.warning(f"Failed to process {file_path}: {e}")
        
        return {
            "status": "success",
            "indexed_files": indexed_count,
            "skipped_files": skipped_count,
            "total_found": len(code_files),
            "languages": list(languages_found),
            "collection": f"{COLLECTION_PREFIX}code"
        }
        
    except Exception as e:
        logger.error(f"Directory indexing failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def code_search(
    query: str,
    language: Optional[str] = None,
    complexity: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Search indexed code using natural language with AST-aware filtering
    
    Args:
        query: Natural language search query
        language: Filter by programming language (python, javascript, etc.)
        complexity: Filter by complexity (low, medium, high)
        limit: Number of results
    """
    try:
        collection_name = f"{COLLECTION_PREFIX}code"
        
        # Build filter conditions
        must_conditions = []
        
        if language:
            must_conditions.append(
                models.FieldCondition(
                    key="language",
                    match=models.MatchValue(value=language.lower())
                )
            )
        
        if complexity:
            must_conditions.append(
                models.FieldCondition(
                    key="ast_patterns",
                    match=models.MatchValue(value=f"complexity:{complexity.lower()}")
                )
            )
        
        # Prepare filter
        query_filter = models.Filter(
            must=must_conditions
        ) if must_conditions else None
        
        # Hybrid search with filters
        dense_embedding = list(dense_model.embed([query]))[0]
        
        semantic_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=("dense", dense_embedding.tolist()),
            query_filter=query_filter,
            limit=limit
        )
        
        results = []
        for hit in semantic_results:
            payload = hit.payload
            results.append({
                "file_path": payload.get("file_path"),
                "language": payload.get("language"),
                "content": payload.get("content"),
                "score": hit.score,
                "complexity_score": payload.get("complexity_score"),
                "function_count": payload.get("function_count"),
                "class_count": payload.get("class_count"),
                "ast_patterns": payload.get("ast_patterns", [])[:10],  # Limit patterns shown
                "analysis": payload.get("analysis", {})
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Code search failed: {e}")
        return []

@mcp.tool()
async def code_find_similar(
    file_path: str,
    similarity_threshold: float = 0.7,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Find files similar to the given file based on AST structure
    
    Args:
        file_path: Reference file path
        similarity_threshold: Minimum similarity score (0-1)
        limit: Number of similar files to return
    """
    try:
        if not ast_analyzer:
            return []
        
        # Analyze reference file
        result = await code_analyze_file(file_path, store_analysis=False)
        if result["status"] != "success":
            return []
        
        ref_analysis = result["analysis"]
        
        # Generate search query from file characteristics
        search_terms = []
        search_terms.append(f"language:{ref_analysis['language']}")
        
        if ref_analysis["functions"]:
            search_terms.append("functions")
        if ref_analysis["classes"]:
            search_terms.append("classes")
        
        query = " ".join(search_terms)
        
        # Search for similar files
        results = await code_search(
            query=query,
            language=ref_analysis["language"],
            limit=limit * 2  # Get more results to filter
        )
        
        # Filter by similarity and exclude self
        similar_files = []
        for result in results:
            if result["file_path"] == file_path:
                continue
            
            # Simple similarity based on structure
            similarity = calculate_structure_similarity(ref_analysis, result["analysis"])
            
            if similarity >= similarity_threshold:
                result["similarity_score"] = similarity
                similar_files.append(result)
        
        # Sort by similarity and limit
        similar_files.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar_files[:limit]
        
    except Exception as e:
        logger.error(f"Similar file search failed: {e}")
        return []

def calculate_structure_similarity(analysis1: Dict, analysis2: Dict) -> float:
    """Calculate similarity between two code analyses"""
    try:
        # Simple similarity based on function names, classes, complexity
        score = 0.0
        
        func_names1 = {f["name"] for f in analysis1.get("functions", [])}
        func_names2 = {f["name"] for f in analysis2.get("functions", [])}
        
        class_names1 = {c["name"] for c in analysis1.get("classes", [])}
        class_names2 = {c["name"] for c in analysis2.get("classes", [])}
        
        # Function similarity
        if func_names1 or func_names2:
            func_intersection = len(func_names1 & func_names2)
            func_union = len(func_names1 | func_names2)
            score += 0.4 * (func_intersection / func_union) if func_union > 0 else 0
        
        # Class similarity  
        if class_names1 or class_names2:
            class_intersection = len(class_names1 & class_names2)
            class_union = len(class_names1 | class_names2)
            score += 0.3 * (class_intersection / class_union) if class_union > 0 else 0
        
        # Complexity similarity
        comp1 = analysis1.get("complexity_score", 0)
        comp2 = analysis2.get("complexity_score", 0)
        if comp1 > 0 or comp2 > 0:
            max_comp = max(comp1, comp2)
            min_comp = min(comp1, comp2)
            score += 0.3 * (min_comp / max_comp) if max_comp > 0 else 0
        
        return min(score, 1.0)
        
    except Exception:
        return 0.0

@mcp.tool()
async def project_create(
    project_path: str = ".",
    project_name: Optional[str] = None,
    auto_index: bool = True
) -> Dict[str, Any]:
    """Create isolated neural container for any project directory
    
    Args:
        project_path: Path to project directory (default: current)
        project_name: Custom project name (auto-detected if not provided)
        auto_index: Automatically index code files after creation
    """
    try:
        import os
        import subprocess
        import socket
        from pathlib import Path
        
        # Resolve absolute project path
        abs_project_path = Path(project_path).resolve()
        if not abs_project_path.exists():
            return {"status": "error", "message": f"Project path not found: {project_path}"}
        
        # Auto-detect project name intelligently
        if not project_name:
            # Try git repo name first
            try:
                git_root = subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"], 
                    cwd=abs_project_path, 
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                project_name = Path(git_root).name
            except:
                # Fall back to directory name
                project_name = abs_project_path.name
        
        # Clean project name for Docker compatibility
        import re
        project_name = re.sub(r'[^a-zA-Z0-9._-]', '', project_name.lower())
        if not project_name:
            project_name = f"project-{int(time.time())}"
        
        container_name = f"l9-{project_name}"
        
        # Check if container already exists
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            if result.stdout.strip():
                # Container exists - check if running
                result = subprocess.run(
                    ["docker", "ps", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
                    capture_output=True, text=True, check=True
                )
                if result.stdout.strip():
                    # Get port
                    port_result = subprocess.run(
                        ["docker", "port", container_name, "6333/tcp"],
                        capture_output=True, text=True
                    )
                    port = port_result.stdout.split(":")[-1].strip() if port_result.returncode == 0 else "6333"
                    return {
                        "status": "exists",
                        "message": f"Neural container already running for {project_name}",
                        "project_name": project_name,
                        "container_name": container_name,
                        "port": int(port),
                        "project_path": str(abs_project_path)
                    }
                else:
                    # Container exists but stopped - restart it
                    subprocess.run(["docker", "start", container_name], check=True)
                    port_result = subprocess.run(
                        ["docker", "port", container_name, "6333/tcp"],
                        capture_output=True, text=True
                    )
                    port = port_result.stdout.split(":")[-1].strip() if port_result.returncode == 0 else "6333"
                    return {
                        "status": "restarted",
                        "message": f"Neural container restarted for {project_name}",
                        "project_name": project_name,
                        "container_name": container_name,
                        "port": int(port),
                        "project_path": str(abs_project_path)
                    }
        except subprocess.CalledProcessError:
            pass  # Container doesn't exist, continue with creation
        
        # Find available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        
        # Create data directories
        data_dir = Path.home() / ".neural-flow" / "data" / project_name
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "qdrant").mkdir(exist_ok=True)
        (data_dir / "mcp").mkdir(exist_ok=True)
        
        # Start new container
        docker_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--restart", "unless-stopped",
            "-p", f"{port}:6333",
            "-p", f"{port+1}:6334",
            "-v", f"{abs_project_path}:/app/project:ro",
            "-v", f"{data_dir}/qdrant:/qdrant/storage",
            "-v", f"{data_dir}/mcp:/app/data",
            "-e", f"PROJECT_NAME={project_name}",
            "-e", "QDRANT_HOST=localhost",
            "-e", "QDRANT_HTTP_PORT=6333",
            "-e", "QDRANT_GRPC_PORT=6334",
            "--label", f"l9.project={project_name}",
            "--label", "l9.type=neural-memory",
            "l9-mcp-server:latest"
        ]
        
        subprocess.run(docker_cmd, check=True)
        
        # Wait for container health check
        import time
        for _ in range(30):
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/collections", timeout=2)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(2)
        
        result = {
            "status": "created",
            "message": f"Neural container created for {project_name}",
            "project_name": project_name,
            "container_name": container_name,
            "port": port,
            "project_path": str(abs_project_path),
            "mcp_config": {
                "server_name": f"l9-neural-{project_name}",
                "command": "docker",
                "args": ["exec", "-i", container_name, "python3", "/app/mcp_server.py"],
                "env": {
                    "PROJECT_NAME": project_name,
                    "QDRANT_HOST": "localhost", 
                    "QDRANT_HTTP_PORT": str(port)
                }
            }
        }
        
        # Auto-index if requested
        if auto_index:
            # Connect to new container and index
            new_qdrant_client = QdrantClient(host="localhost", port=port)
            new_dense_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
            
            indexed_count = 0
            for file_path in abs_project_path.rglob("*"):
                if file_path.suffix in [".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"]:
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if content.strip():
                            # Simple chunking
                            embedding = list(new_dense_model.embed([content[:2000]]))[0]
                            new_qdrant_client.upsert(
                                collection_name=f"project_{project_name}_code",
                                points=[{
                                    "id": hash(str(file_path)) % (10**8),
                                    "vector": {"dense": embedding.tolist()},
                                    "payload": {
                                        "content": content[:2000],
                                        "file_path": str(file_path.relative_to(abs_project_path)),
                                        "language": file_path.suffix[1:],
                                        "timestamp": datetime.now().isoformat()
                                    }
                                }]
                            )
                            indexed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to index {file_path}: {e}")
            
            result["indexed_files"] = indexed_count
        
        return result
        
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Docker command failed: {e}"}
    except Exception as e:
        logger.error(f"Project creation failed: {e}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def project_list() -> Dict[str, Any]:
    """List all neural memory containers"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "label=l9.type=neural-memory", 
             "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Label \"l9.project\"}}"],
            capture_output=True, text=True, check=True
        )
        
        projects = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('\t')
                if len(parts) >= 4:
                    name, status, ports, project_name = parts
                    port = ""
                    if ":" in ports:
                        port = ports.split(":")[1].split("->")[0] if "->" in ports else ""
                    
                    projects.append({
                        "project_name": project_name,
                        "container_name": name,
                        "status": status,
                        "port": port,
                        "running": "Up" in status
                    })
        
        return {"status": "success", "projects": projects}
        
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to list containers: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def project_stop(project_name: str) -> Dict[str, Any]:
    """Stop neural container for a specific project"""
    try:
        container_name = f"l9-{project_name}"
        subprocess.run(["docker", "stop", container_name], check=True)
        return {
            "status": "success", 
            "message": f"Stopped neural container for {project_name}"
        }
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Failed to stop container: {e}"}

@mcp.tool()
async def collection_stats() -> Dict[str, Any]:
    """Get statistics for all collections"""
    try:
        collections = qdrant_client.get_collections().collections
        stats = {}
        
        for collection in collections:
            if collection.name.startswith(COLLECTION_PREFIX):
                info = qdrant_client.get_collection(collection.name)
                stats[collection.name] = {
                    "vectors_count": info.vectors_count,
                    "points_count": info.points_count,
                    "status": info.status
                }
        
        return {
            "project": PROJECT_NAME,
            "collections": stats,
            "total_points": sum(s["points_count"] for s in stats.values()),
            "tree_sitter_available": ast_analyzer is not None,
            "supported_languages": list(ast_analyzer.extension_map.values()) if ast_analyzer else []
        }
        
    except Exception as e:
        logger.error(f"Stats failed: {e}")
        return {"status": "error", "message": str(e)}

# Run the server
if __name__ == "__main__":
    asyncio.run(initialize())
    mcp.run(transport='stdio')