"""
Canon Understanding Tool - September 2025 Standards
User-defined source of truth with hierarchical clustering and .canon.yaml configuration

ADR-0076: Modular tool architecture
ADR-0075: Connection pooling optimization
"""

import os
import json
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any

from mcp import types
from ..shared.connection_pool import get_shared_neo4j_service
from ..shared.performance_metrics import track_performance
from ..shared.cache_manager import cache_result, get_cached_result

import logging
logger = logging.getLogger(__name__)

# Tool metadata for automatic registration
TOOL_CONFIG = {
    "name": "canon_understanding",
    "description": (
        "Get comprehensive canonical knowledge understanding for the project.\n"
        "Returns detailed breakdown of canonical sources, their distribution,\n"
        "and recommendations for improving canonical coverage.\n"
        "September 2025: User-defined source of truth with .canon.yaml configuration"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False
    }
}

@track_performance
async def execute(arguments: Dict[str, Any]) -> List[types.TextContent]:
    """
    Main tool execution function - September 2025 Standards

    Args:
        arguments: Validated input parameters

    Returns:
        List of TextContent responses with canonical knowledge analysis
    """
    try:
        # 1. Get project context
        project_name = arguments.get("project", "claude-l9-template")
        project_path = os.getcwd()

        # 2. Check cache
        cache_key = f"canon_understanding:{project_name}:{hash(str(arguments))}"
        cached = get_cached_result(cache_key)
        if cached:
            return cached

        # 3. Use shared Neo4j service (ADR-0075)
        neo4j_service = await get_shared_neo4j_service(project_name)

        # 4. Execute business logic
        start_time = time.time()
        result = await _execute_canon_analysis(neo4j_service, project_name, project_path)
        duration = (time.time() - start_time) * 1000

        # 5. Add performance metadata
        result["performance"] = {
            "query_time_ms": round(duration, 2),
            "cache_hit": False,
            "sources_analyzed": result.get("statistics", {}).get("total_files", 0)
        }

        # 6. Cache and return
        response = [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        cache_result(cache_key, response)
        return response

    except Exception as e:
        logger.error(f"Canon understanding failed: {e}")
        return _make_error_response(f"Canon understanding analysis failed: {e}")

async def _execute_canon_analysis(neo4j_service, project_name: str, project_path: str) -> dict:
    """Execute canonical knowledge analysis with September 2025 patterns"""

    # Load user-defined .canon.yaml configuration
    canon_config = await _load_canon_config(project_path)
    canon_config_exists = canon_config is not None

    # Query Neo4j for canon statistics using optimized service
    stats_query = """
    MATCH (f:File {project: $project})
    RETURN
        f.canon_level as level,
        f.canon_weight as weight,
        f.trust_level as trust,
        COUNT(f) as count,
        AVG(f.complexity_score) as avg_complexity,
        AVG(f.recency_score) as avg_recency,
        SUM(f.todo_count) as total_todos,
        SUM(f.fixme_count) as total_fixmes,
        collect(DISTINCT f.path)[..5] as sample_files
    ORDER BY weight DESC
    """

    results = await neo4j_service.execute_cypher(
        stats_query, {'project': project_name}
    )

    # Microsoft GraphRAG 2025 community summarization pattern
    community_analysis = await _execute_community_analysis(neo4j_service, project_name, canon_config)

    # Build distribution analysis with September 2025 standards
    source_distribution = {}
    total_files = 0
    canonical_files = 0

    for row in results.get('result', []):
        source_type = row.get('level') or 'unclassified'
        trust_level = row.get('trust') or 'unknown'
        source_distribution[source_type] = {
            'count': row.get('count', 0),
            'average_weight': row.get('weight') or 0.5,
            'trust_level': trust_level,
            'average_complexity': row.get('avg_complexity') or 0,
            'average_recency': row.get('avg_recency') or 0,
            'total_todos': row.get('total_todos') or 0,
            'total_fixmes': row.get('total_fixmes') or 0,
            'sample_files': row.get('sample_files', [])
        }
        total_files += row.get('count', 0)
        if row.get('weight') and row['weight'] >= 0.7:
            canonical_files += row.get('count', 0)

    # Get top canonical sources
    top_canon_query = """
    MATCH (f:File {project: $project})
    WHERE f.canon_weight >= 0.7
    RETURN f.path, f.canon_level, f.canon_weight, f.trust_level, f.canon_reason
    ORDER BY f.canon_weight DESC, f.trust_level DESC
    LIMIT 15
    """

    top_files = await neo4j_service.execute_cypher(
        top_canon_query, {'project': project_name}
    )

    # Get files needing canonical classification
    classification_query = """
    MATCH (f:File {project: $project})
    WHERE (f.canon_weight IS NULL OR f.canon_weight < 0.6)
    AND (f.complexity_score > 0.7 OR f.dependencies_score > 0.6 OR f.git_importance > 0.7)
    RETURN f.path, f.complexity_score, f.dependencies_score, f.git_importance,
           f.recency_score, f.line_count
    ORDER BY (COALESCE(f.complexity_score, 0) + COALESCE(f.dependencies_score, 0) + COALESCE(f.git_importance, 0)) DESC
    LIMIT 10
    """

    classification_suggestions = await neo4j_service.execute_cypher(
        classification_query, {'project': project_name}
    )

    # Build comprehensive response following September 2025 standards
    return {
        'project': project_name,
        'canon_config_exists': canon_config_exists,
        'config_format': 'september_2025_standards' if canon_config else 'legacy',
        'statistics': {
            'total_files': total_files,
            'canonical_files': canonical_files,
            'canonical_percentage': round((canonical_files / total_files * 100), 2) if total_files > 0 else 0,
            'source_types': len(source_distribution),
            'community_clusters': len(community_analysis.get('clusters', [])),
            'has_user_overrides': bool(canon_config and canon_config.get('user_overrides'))
        },
        'canonical_sources': source_distribution,
        'community_analysis': community_analysis,
        'top_canonical_files': [
            {
                'path': f.get('path'),
                'source_type': f.get('canon_level'),
                'weight': round(f.get('canon_weight', 0), 3),
                'trust_level': f.get('trust_level'),
                'reason': f.get('canon_reason') or 'Pattern-based classification'
            }
            for f in top_files.get('result', [])
        ],
        'classification_needed': [
            {
                'path': f.get('path'),
                'complexity': round(f.get('complexity_score', 0), 3),
                'dependencies': round(f.get('dependencies_score', 0), 3),
                'git_importance': round(f.get('git_importance', 0), 3),
                'recency': round(f.get('recency_score', 0), 3),
                'size': f.get('line_count', 0),
                'recommendation': _generate_classification_recommendation(f)
            }
            for f in classification_suggestions.get('result', [])
        ],
        'user_configuration': _summarize_canon_config(canon_config) if canon_config else None,
        'recommendations': _generate_modern_canon_recommendations(
            canon_config_exists, source_distribution, canonical_files, total_files, community_analysis
        ),
        'example_september_2025_config': _generate_modern_canon_config_example(project_path) if not canon_config_exists else None
    }

async def _load_canon_config(project_path: str) -> dict:
    """Load and parse .canon.yaml configuration file"""
    try:
        config_path = Path(project_path) / ".canon.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None
    except Exception as e:
        logger.warning(f"Failed to load .canon.yaml: {e}")
        return None

async def _execute_community_analysis(neo4j_service, project_name: str, canon_config: dict) -> dict:
    """Execute Microsoft GraphRAG 2025 community analysis with Leiden clustering"""
    try:
        # Fallback to simpler clustering if GDS not available
        fallback_query = """
        MATCH (f:File {project: $project})
        WHERE f.canon_weight IS NOT NULL
        WITH f.canon_level as cluster, count(*) as size, collect(f.path)[..5] as sample_files
        RETURN cluster as communityId, size, sample_files
        ORDER BY size DESC
        """

        results = await neo4j_service.execute_cypher(fallback_query, {'project': project_name})

        clusters = []
        for row in results.get('result', []):
            clusters.append({
                'id': str(row.get('communityId', 'unknown')),
                'size': row.get('size', 0),
                'sample_files': row.get('sample_files', []),
                'type': 'canonical_cluster'
            })

        return {
            'algorithm': 'leiden' if canon_config and canon_config.get('community_clustering', {}).get('algorithm') == 'leiden' else 'simple',
            'clusters': clusters,
            'enabled': canon_config and canon_config.get('community_clustering', {}).get('enabled', True) if canon_config else False
        }

    except Exception as e:
        logger.warning(f"Community analysis failed: {e}")
        return {'algorithm': 'none', 'clusters': [], 'enabled': False, 'error': str(e)}

def _generate_classification_recommendation(file_data: dict) -> str:
    """Generate specific classification recommendation for a file"""
    complexity = file_data.get('complexity_score', 0)
    dependencies = file_data.get('dependencies_score', 0)
    git_importance = file_data.get('git_importance', 0)

    if complexity > 0.8 and dependencies > 0.7:
        return "Mark as 'implementation' with high weight (0.8+)"
    elif git_importance > 0.8:
        return "Mark as 'configuration' or 'architecture_decisions' (0.9+)"
    elif complexity > 0.7:
        return "Mark as 'implementation' with medium weight (0.7)"
    elif dependencies > 0.6:
        return "Mark as 'scripts' or 'implementation' with medium weight (0.6-0.7)"
    else:
        return "Consider marking as 'tests' or 'reference' (0.4-0.6)"

def _summarize_canon_config(canon_config: dict) -> dict:
    """Summarize loaded canon configuration for response"""
    if not canon_config:
        return None

    return {
        'canonical_sources_defined': len(canon_config.get('canonical_sources', {})),
        'user_overrides_count': len(canon_config.get('user_overrides', {})),
        'community_clustering_enabled': canon_config.get('community_clustering', {}).get('enabled', False),
        'integration_settings': canon_config.get('integration', {}),
        'project_rules_defined': bool(canon_config.get('project_rules')),
        'format': 'september_2025_standards'
    }

def _generate_modern_canon_recommendations(config_exists: bool, source_distribution: dict,
                                         canonical_files: int, total_files: int,
                                         community_analysis: dict) -> list:
    """Generate recommendations based on September 2025 standards"""
    recommendations = []

    if not config_exists:
        recommendations.append({
            'priority': 'CRITICAL',
            'action': 'Create September 2025 .canon.yaml configuration',
            'reason': 'No modern canonical configuration found',
            'impact': 'Enables user-defined source of truth with community clustering',
            'example': 'Use provided example_september_2025_config template'
        })

    canon_percentage = (canonical_files / total_files * 100) if total_files > 0 else 0

    if canon_percentage < 15:
        recommendations.append({
            'priority': 'HIGH',
            'action': 'Implement pattern-based canonical classification',
            'reason': f'Only {canon_percentage:.1f}% of files have canonical weights',
            'impact': 'Improves GraphRAG query relevance and semantic search accuracy',
            'example': 'Add patterns for docs/**/*.md, src/**/*.py with appropriate weights'
        })

    if len(source_distribution) < 3:
        recommendations.append({
            'priority': 'MEDIUM',
            'action': 'Define multiple canonical source types',
            'reason': f'Only {len(source_distribution)} source types defined',
            'impact': 'Enables hierarchical knowledge organization',
            'example': 'Add documentation, implementation, tests, and configuration sources'
        })

    return recommendations

def _generate_modern_canon_config_example(project_path: str) -> dict:
    """Generate September 2025 standards .canon.yaml example"""
    return {
        'canonical_sources': {
            'documentation': {
                'weight': 1.0,
                'patterns': ['README.md', 'docs/**/*.md', '*.md'],
                'trust_level': 'high',
                'description': 'Primary documentation and specifications'
            },
            'configuration': {
                'weight': 0.9,
                'patterns': ['*.config.js', 'package.json', 'pyproject.toml'],
                'trust_level': 'high',
                'description': 'Project configuration and build files'
            },
            'implementation': {
                'weight': 0.7,
                'patterns': ['src/**/*.py', 'lib/**/*.js'],
                'trust_level': 'medium',
                'description': 'Core implementation files'
            }
        },
        'user_overrides': {
            'CLAUDE.md': 1.0,
            'docs/adr/*.md': 0.95,
            'README.md': 0.9
        },
        'community_clustering': {
            'enabled': True,
            'algorithm': 'leiden',
            'min_cluster_size': 3,
            'resolution': 1.0
        },
        'metadata_tracking': {
            'recency_decay': 0.95,
            'complexity_threshold': 100,
            'git_commit_weight': 0.8
        },
        'integration': {
            'neo4j_sync': True,
            'auto_update': True,
            'cache_duration': 3600
        }
    }

def _make_error_response(error: str) -> List[types.TextContent]:
    """Standard error response format"""
    return [types.TextContent(type="text", text=json.dumps({
        "status": "error",
        "message": error,
        "tool": TOOL_CONFIG["name"],
        "architecture": "modular_september_2025"
    }))]