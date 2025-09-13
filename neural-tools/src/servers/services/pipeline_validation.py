#!/usr/bin/env python3
"""
Pipeline Validation Service - ADR-0034 Implementation
Provides validation checkpoints at each stage of the project pipeline

This service implements the validation functions identified in ADR-0034
to ensure project synchronization across detection, orchestration, storage, and search.

Author: L9 Engineering Team
Date: 2025-09-12
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PipelineValidationService:
    """
    Validation service for project pipeline synchronization
    Implements ADR-0034 validation checkpoints
    """
    
    def __init__(self):
        self.validation_results: List[Dict[str, Any]] = []
    
    async def validate_project_detection(self, project_name: str, project_path: str, confidence: float) -> bool:
        """
        Validate project detection results
        
        Args:
            project_name: Detected project name
            project_path: Detected project path
            confidence: Detection confidence score
            
        Returns:
            True if validation passes, False otherwise
        """
        validation_start = datetime.now()
        issues = []
        
        logger.info(f"🔍 [Validation] Starting project detection validation")
        logger.info(f"🔍 [Validation] Project: {project_name}, Path: {project_path}, Confidence: {confidence}")
        
        # Check 1: Project name should not be default for valid projects
        if project_name == "default" and confidence > 0.5:
            issues.append("High confidence detection returned 'default' - possible detection failure")
            logger.warning(f"⚠️ [Validation] Project detection issue: {issues[-1]}")
        
        # Check 2: Project path must exist
        if not Path(project_path).exists():
            issues.append(f"Project path does not exist: {project_path}")
            logger.error(f"❌ [Validation] Project path validation failed: {issues[-1]}")
            return False
        
        # Check 3: Confidence should be reasonable for valid projects
        if confidence < 0.3:
            issues.append(f"Very low confidence detection: {confidence}")
            logger.warning(f"⚠️ [Validation] Low confidence: {issues[-1]}")
        
        # Check 4: Project name should be sanitized (no special chars except hyphens)
        import re
        if not re.match(r'^[a-z0-9-]+$', project_name) and project_name != "default":
            issues.append(f"Project name contains invalid characters: {project_name}")
            logger.warning(f"⚠️ [Validation] Invalid project name: {issues[-1]}")
        
        # Record validation result
        result = {
            "stage": "project_detection",
            "timestamp": validation_start.isoformat(),
            "project_name": project_name,
            "project_path": project_path,
            "confidence": confidence,
            "issues": issues,
            "passed": len(issues) == 0 or (len(issues) == 1 and "low confidence" in issues[0].lower())
        }
        self.validation_results.append(result)
        
        if result["passed"]:
            logger.info(f"✅ [Validation] Project detection validation passed")
        else:
            logger.error(f"❌ [Validation] Project detection validation failed: {issues}")
        
        return result["passed"]
    
    async def validate_container_sync(self, container_name: str, project_name: str, mounted_path: str) -> bool:
        """
        Validate container name matches project
        
        Args:
            container_name: Name of the spawned container
            project_name: Detected project name
            mounted_path: Path mounted in container
            
        Returns:
            True if validation passes, False otherwise
        """
        validation_start = datetime.now()
        issues = []
        
        logger.info(f"🔍 [Validation] Starting container sync validation")
        logger.info(f"🔍 [Validation] Container: {container_name}, Project: {project_name}, Mount: {mounted_path}")
        
        # Check 1: Container name should match pattern
        expected_name = f"indexer-{project_name}"
        if container_name != expected_name:
            issues.append(f"Container name mismatch: {container_name} != {expected_name}")
            logger.error(f"❌ [Validation] Container name mismatch: {issues[-1]}")
        
        # Check 2: Mounted path should exist
        if not Path(mounted_path).exists():
            issues.append(f"Mounted path does not exist: {mounted_path}")
            logger.error(f"❌ [Validation] Mount path validation failed: {issues[-1]}")
        
        # Check 3: Project name consistency
        if project_name not in container_name:
            issues.append(f"Project name not found in container name: {project_name} not in {container_name}")
            logger.error(f"❌ [Validation] Project name consistency failed: {issues[-1]}")
        
        # Record validation result
        result = {
            "stage": "container_sync",
            "timestamp": validation_start.isoformat(),
            "container_name": container_name,
            "project_name": project_name,
            "mounted_path": mounted_path,
            "expected_name": expected_name,
            "issues": issues,
            "passed": len(issues) == 0
        }
        self.validation_results.append(result)
        
        if result["passed"]:
            logger.info(f"✅ [Validation] Container sync validation passed")
        else:
            logger.error(f"❌ [Validation] Container sync validation failed: {issues}")
        
        return result["passed"]
    
    async def validate_collection_naming(self, collection_name: str, project_name: str, collection_type: str = "qdrant") -> bool:
        """
        Validate collection name follows standard
        
        Args:
            collection_name: Name of the collection/database
            project_name: Detected project name
            collection_type: Type of collection ("qdrant" or "neo4j")
            
        Returns:
            True if validation passes, False otherwise
        """
        validation_start = datetime.now()
        issues = []
        
        logger.info(f"🔍 [Validation] Starting collection naming validation")
        logger.info(f"🔍 [Validation] Collection: {collection_name}, Project: {project_name}, Type: {collection_type}")
        
        if collection_type == "qdrant":
            expected_name = f"project-{project_name}"
            if collection_name != expected_name:
                issues.append(f"Qdrant collection name mismatch: {collection_name} != {expected_name}")
                logger.error(f"❌ [Validation] Qdrant collection naming failed: {issues[-1]}")
            
            # Check for old _code suffix
            if "_code" in collection_name:
                issues.append(f"Collection contains deprecated _code suffix: {collection_name}")
                logger.error(f"❌ [Validation] Deprecated suffix found: {issues[-1]}")
        
        elif collection_type == "neo4j":
            expected_name = f"project_{project_name.replace('-', '_')}"
            if collection_name != expected_name:
                issues.append(f"Neo4j database name mismatch: {collection_name} != {expected_name}")
                logger.error(f"❌ [Validation] Neo4j database naming failed: {issues[-1]}")
        
        # Record validation result
        result = {
            "stage": "collection_naming",
            "timestamp": validation_start.isoformat(),
            "collection_name": collection_name,
            "project_name": project_name,
            "collection_type": collection_type,
            "issues": issues,
            "passed": len(issues) == 0
        }
        self.validation_results.append(result)
        
        if result["passed"]:
            logger.info(f"✅ [Validation] Collection naming validation passed")
        else:
            logger.error(f"❌ [Validation] Collection naming validation failed: {issues}")
        
        return result["passed"]
    
    async def validate_end_to_end_sync(self, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate end-to-end project synchronization
        
        Args:
            project_info: Project information from ProjectContextManager
            
        Returns:
            Validation result dictionary
        """
        validation_start = datetime.now()
        logger.info(f"🔍 [Validation] Starting end-to-end synchronization validation")
        
        project_name = project_info.get("project", "unknown")
        project_path = project_info.get("path", "")
        confidence = project_info.get("confidence", 0.0)
        
        # Run all validation checks
        results = {
            "validation_timestamp": validation_start.isoformat(),
            "project_info": project_info,
            "checks": {}
        }
        
        # 1. Project detection validation
        detection_passed = await self.validate_project_detection(project_name, project_path, confidence)
        results["checks"]["project_detection"] = detection_passed
        
        # 2. Collection naming validation
        qdrant_collection = f"project-{project_name}"
        neo4j_database = f"project_{project_name.replace('-', '_')}"
        
        qdrant_passed = await self.validate_collection_naming(qdrant_collection, project_name, "qdrant")
        neo4j_passed = await self.validate_collection_naming(neo4j_database, project_name, "neo4j")
        
        results["checks"]["qdrant_naming"] = qdrant_passed
        results["checks"]["neo4j_naming"] = neo4j_passed
        
        # 3. Overall validation result
        all_passed = all(results["checks"].values())
        results["overall_passed"] = all_passed
        results["total_checks"] = len(results["checks"])
        results["passed_checks"] = sum(results["checks"].values())
        
        if all_passed:
            logger.info(f"✅ [Validation] End-to-end validation PASSED ({results['passed_checks']}/{results['total_checks']} checks)")
        else:
            logger.error(f"❌ [Validation] End-to-end validation FAILED ({results['passed_checks']}/{results['total_checks']} checks)")
        
        return results
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get all validation results from this session"""
        return self.validation_results.copy()
    
    def clear_validation_history(self):
        """Clear validation history"""
        self.validation_results.clear()
        logger.info("🔄 [Validation] Cleared validation history")


# Global validation service instance
validation_service = PipelineValidationService()


async def validate_pipeline_stage(stage: str, **kwargs) -> bool:
    """
    Convenience function for pipeline stage validation
    
    Args:
        stage: Stage name ("detection", "container", "collection", "end_to_end")
        **kwargs: Stage-specific parameters
        
    Returns:
        True if validation passes, False otherwise
    """
    if stage == "detection":
        return await validation_service.validate_project_detection(
            kwargs.get("project_name", ""),
            kwargs.get("project_path", ""),
            kwargs.get("confidence", 0.0)
        )
    elif stage == "container":
        return await validation_service.validate_container_sync(
            kwargs.get("container_name", ""),
            kwargs.get("project_name", ""),
            kwargs.get("mounted_path", "")
        )
    elif stage == "collection":
        return await validation_service.validate_collection_naming(
            kwargs.get("collection_name", ""),
            kwargs.get("project_name", ""),
            kwargs.get("collection_type", "qdrant")
        )
    elif stage == "end_to_end":
        return await validation_service.validate_end_to_end_sync(
            kwargs.get("project_info", {})
        )
    else:
        logger.error(f"❌ [Validation] Unknown validation stage: {stage}")
        return False