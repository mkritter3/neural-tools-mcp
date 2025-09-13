#!/usr/bin/env python3
"""
Neural Search API Server
FastAPI-based REST API for indexing and searching codebases
Uses existing AsyncQdrantClient + multi-tenant architecture
"""

import logging
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import our existing services
import sys
sys.path.append(str(Path(__file__).parent.parent))

from servers.services.qdrant_service import QdrantService
# Authentication disabled for development
# from api.auth import get_api_key, get_auth_summary
from servers.services.collection_config import CollectionManager, CollectionType
from api.file_watcher import AsyncProjectWatcher
from api.code_chunker import SmartCodeChunker
from api.config_manager import ConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances (initialized in lifespan)
qdrant_service: Optional[QdrantService] = None
collection_manager: Optional[CollectionManager] = None
file_watcher: Optional[AsyncProjectWatcher] = None
code_chunker: Optional[SmartCodeChunker] = None
config_manager: Optional[ConfigManager] = None

# In-memory job store for solo developer simplicity
class Job(BaseModel):
    status: str = Field(..., pattern="^(queued|running|completed|failed)$")
    project_name: str
    details: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

job_store: Dict[str, Job] = {}

# Request/Response models
class IndexRequest(BaseModel):
    project_path: str = Field(..., description="Absolute path to project directory")
    project_name: str = Field(..., description="Unique project identifier")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    project_names: Optional[List[str]] = Field(None, description="Projects to search (null = all)")
    limit: int = Field(10, ge=1, le=50)

class SearchResult(BaseModel):
    file_path: str
    project_name: str
    score: float
    snippet: str

# File change handler for automatic re-indexing
async def handle_file_change(project_name: str, file_path: str):
    """Handle file changes by triggering re-indexing"""
    logger.info(f"🔄 File change detected in {project_name}: {Path(file_path).name}")
    
    try:
        # Create a background job for re-indexing this file
        job_id = str(uuid.uuid4())
        job_store[job_id] = Job(
            status="queued",
            project_name=project_name,
            details=f"Re-indexing due to file change: {Path(file_path).name}"
        )
        
        # For now, trigger full project re-index (TODO: optimize to single file)
        # In production, we'd implement incremental indexing
        logger.info(f"🚀 Triggered re-index job {job_id} for project {project_name}")
        
    except Exception as e:
        logger.error(f"❌ Failed to handle file change: {e}")

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, cleanup on shutdown"""
    global qdrant_service, collection_manager, file_watcher, code_chunker, config_manager
    
    logger.info("🚀 Starting Neural Search API Server")
    
    # Initialize services
    try:
        # Initialize configuration manager first
        config_manager = ConfigManager()
        logger.info(f"📋 Configuration: {config_manager.get_config_summary()}")
        
        # Authentication disabled for development
        logger.info("🔓 Authentication: Disabled for development")
        
        # Use default project for service initialization
        qdrant_service = QdrantService("neural-api")
        init_result = await qdrant_service.initialize()
        
        if not init_result.get("success"):
            logger.error(f"Failed to initialize Qdrant: {init_result}")
            raise RuntimeError("Qdrant initialization failed")
        
        collection_manager = CollectionManager("neural-api")
        
        # Initialize file watcher
        file_watcher = AsyncProjectWatcher()
        file_watcher.initialize(handle_file_change)
        
        # Initialize code chunker
        code_chunker = SmartCodeChunker()
        
        # Auto-start watching configured projects
        auto_watch_projects = config_manager.get_projects_for_auto_watch()
        for project_name, project_path in auto_watch_projects.items():
            try:
                logger.info(f"🔄 Auto-starting watch for project '{project_name}'")
                success = await file_watcher.start_watching_project(project_name, project_path)
                if success:
                    logger.info(f"👀 Auto-watching project '{project_name}' at {project_path}")
                else:
                    logger.warning(f"⚠️  Failed to auto-watch project '{project_name}'")
            except Exception as e:
                logger.error(f"❌ Failed to auto-watch project '{project_name}': {e}")
        
        logger.info("✅ Services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("🛑 Shutting down Neural Search API Server")
    if file_watcher:
        try:
            await file_watcher.stop_all()
            logger.info("✅ File watchers stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping file watchers: {e}")
        
        file_watcher.shutdown()

# Create FastAPI app with lifespan management
app = FastAPI(
    title="Neural Search API",
    description="Local neural search for solo developers using AsyncQdrantClient + Multi-tenant architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Basic indexing implementation (blocking for now)
async def perform_indexing(project_path: str, project_name: str) -> Dict[str, Any]:
    """Core indexing logic - discovers files, creates embeddings, stores vectors"""
    try:
        project_dir = Path(project_path)
        if not project_dir.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # Initialize project-specific collection manager
        project_collection_manager = CollectionManager(project_name)
        
        # Get the code collection name using existing multi-tenant pattern
        code_collection_name = project_collection_manager.get_collection_name(CollectionType.CODE)
        logger.info(f"Indexing into collection: {code_collection_name}")
        
        # Ensure collection exists
        success = await qdrant_service.ensure_collection(code_collection_name, 768)
        if not success:
            raise RuntimeError(f"Failed to create collection: {code_collection_name}")
        
        # File discovery - find all supported code files
        supported_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.go', '.rs', '.java', '.cpp', '.c', '.h']
        code_files = []
        
        for ext in supported_extensions:
            code_files.extend(list(project_dir.rglob(f"*{ext}")))
        
        # Filter out common directories to ignore
        ignore_patterns = {".venv", "__pycache__", ".git", "node_modules", ".pytest_cache", "target", "build", "dist"}
        code_files = [
            f for f in code_files 
            if not any(ignore_dir in f.parts for ignore_dir in ignore_patterns)
        ]
        
        logger.info(f"Found {len(code_files)} code files to index")
        
        # Smart chunking using AST and regex patterns
        from qdrant_client.models import PointStruct
        
        points = []
        total_chunks = 0
        
        for file_path in code_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Skip empty files
                if not content.strip():
                    continue
                
                # Extract semantic chunks using smart chunker
                global code_chunker
                if code_chunker is None:
                    code_chunker = SmartCodeChunker()
                chunks = code_chunker.chunk_file(str(file_path), content)
                total_chunks += len(chunks)
                
                for chunk in chunks:
                    # Create dummy embedding (768 dimensions with small random values)
                    # TODO: Replace with actual Nomic embeddings
                    import random
                    dummy_embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
                    
                    # Create point with chunk data
                    chunk_dict = chunk.to_dict()
                    
                    point = PointStruct(
                        id=chunk_dict["id"],
                        vector={"dense": dummy_embedding},
                        payload={
                            **chunk_dict["payload"],
                            "project_name": project_name,
                            "content": chunk.text[:1000],  # First 1000 chars as snippet
                            "indexed_at": time.time()
                        }
                    )
                    points.append(point)
                
                logger.debug(f"Extracted {len(chunks)} chunks from {file_path.name}")
                
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")
                continue
        
        # Upsert points to Qdrant
        if points:
            upsert_result = await qdrant_service.upsert_points(code_collection_name, points)
            logger.info(f"Upserted {len(points)} points: {upsert_result}")
        
        # Start watching this project for future changes
        if file_watcher:
            watch_success = await file_watcher.start_watching_project(project_name, project_path)
            if watch_success:
                logger.info(f"👀 Started watching project '{project_name}' for file changes")
            else:
                logger.warning(f"⚠️  Failed to start watching project '{project_name}'")
        
        # Persist project to configuration for auto-restart
        if config_manager:
            config_success = config_manager.add_project(project_name, project_path, auto_watch=True)
            if config_success:
                config_manager.update_project_indexed_time(project_name)
                logger.info(f"💾 Persisted project '{project_name}' to configuration")
            else:
                logger.warning(f"⚠️  Failed to persist project '{project_name}' to configuration")
        
        return {
            "files_processed": len(code_files),
            "chunks_created": total_chunks,
            "points_created": len(points),
            "collection": code_collection_name,
            "watching": file_watcher is not None and watch_success
        }
        
    except Exception as e:
        logger.error(f"Indexing failed for {project_name}: {e}")
        raise

# Job wrapper for BackgroundTasks
async def run_indexing_job(job_id: str, project_path: str, project_name: str):
    """Wrapper that manages job status around core indexing logic"""
    job_store[job_id].status = "running"
    job_store[job_id].updated_at = time.time()
    
    try:
        result = await perform_indexing(project_path, project_name)
        job_store[job_id].status = "completed"
        job_store[job_id].details = f"Processed {result['files_processed']} files, created {result['points_created']} vectors"
        
    except Exception as e:
        job_store[job_id].status = "failed"
        job_store[job_id].details = str(e)
        logger.error(f"Job {job_id} failed: {e}")
    
    finally:
        job_store[job_id].updated_at = time.time()

# API Endpoints
@app.post("/index", status_code=202)
async def index_project(req: IndexRequest, background_tasks: BackgroundTasks):
    """Start indexing a project directory (non-blocking)"""
    job_id = str(uuid.uuid4())
    
    # Create job record
    job_store[job_id] = Job(
        status="queued",
        project_name=req.project_name
    )
    
    # Add to background tasks
    background_tasks.add_task(
        run_indexing_job,
        job_id,
        req.project_path,
        req.project_name
    )
    
    logger.info(f"Started indexing job {job_id} for project {req.project_name}")
    
    return {
        "message": "Indexing job started",
        "job_id": job_id,
        "project_name": req.project_name
    }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an indexing job"""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return job

@app.get("/search")
async def search_projects(
    query: str,
    project_names: Optional[str] = None,  # Comma-separated project names
    limit: int = 10
):
    """Search across one or more projects"""
    try:
        # Parse project names
        if project_names:
            project_list = [name.strip() for name in project_names.split(",")]
        else:
            project_list = None
        
        # Generate dummy query embedding for now
        # TODO: Replace with actual Nomic embeddings
        import random
        query_embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
        
        results = []
        
        if project_list:
            # Search specific projects
            for project_name in project_list:
                project_collection_manager = CollectionManager(project_name)
                collection_name = project_collection_manager.get_collection_name(CollectionType.CODE)
                
                # Search this project's collection
                project_results = await qdrant_service.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )
                
                # Add project context to results
                for result in project_results:
                    result["project_name"] = project_name
                
                results.extend(project_results)
        else:
            # Search all projects - get all collections
            all_collections = await qdrant_service.get_collections()
            code_collections = [c for c in all_collections if c.startswith('project_') and c.endswith('_code')]
            
            for collection_name in code_collections:
                # Extract project name from collection name
                project_name = collection_name.replace('project_', '').replace('_code', '')
                
                project_results = await qdrant_service.search_vectors(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )
                
                # Add project context to results
                for result in project_results:
                    result["project_name"] = project_name
                
                results.extend(project_results)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        results = results[:limit]
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "file_path": result["payload"]["file_path"],
                "project_name": result["project_name"],
                "score": result["score"],
                "snippet": result["payload"]["content"][:200] + "..." if len(result["payload"]["content"]) > 200 else result["payload"]["content"]
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(formatted_results)
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/projects")
async def list_projects():
    """List all indexed projects"""
    try:
        all_collections = await qdrant_service.get_collections()
        code_collections = [c for c in all_collections if c.startswith('project_') and c.endswith('_code')]
        
        projects = []
        for collection_name in code_collections:
            project_name = collection_name.replace('project_', '').replace('_code', '')
            
            # Get collection info
            collection_info = await qdrant_service.get_collection_info(collection_name)
            
            projects.append({
                "project_name": project_name,
                "collection_name": collection_name,
                "points_count": collection_info.get("points_count", 0),
                "status": "indexed" if collection_info.get("exists") else "missing"
            })
        
        return {
            "projects": projects,
            "total_projects": len(projects)
        }
        
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list projects: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        qdrant_health = await qdrant_service.health_check()
        
        return {
            "status": "healthy" if qdrant_health.get("healthy") else "unhealthy",
            "qdrant": qdrant_health,
            "job_store_size": len(job_store)
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )