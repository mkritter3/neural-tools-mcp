#!/usr/bin/env python3
"""
L9 Hook Project Analyzer - Shared Project Analysis Functions
Centralizes project analysis logic used across hooks to eliminate code duplication
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import time
from datetime import datetime, timedelta

from .utilities import find_files_by_pattern, read_file_safely, estimate_tokens


def analyze_project_structure(project_dir: Path) -> Dict[str, Any]:
    """Comprehensive project structure analysis"""
    
    structure = {
        'directories': {},
        'file_types': {},
        'total_files': 0,
        'total_size': 0,
        'depth_analysis': {},
        'organization_score': 0.0
    }
    
    try:
        # Analyze directory structure
        all_files = list(project_dir.rglob('*'))
        
        # Count by directory
        dir_counts = {}
        file_type_counts = {}
        depth_counts = {}
        
        for item in all_files:
            if item.is_file():
                # Directory analysis
                parent_name = item.parent.name
                dir_counts[parent_name] = dir_counts.get(parent_name, 0) + 1
                
                # File type analysis
                suffix = item.suffix.lower() if item.suffix else 'no_extension'
                file_type_counts[suffix] = file_type_counts.get(suffix, 0) + 1
                
                # Depth analysis
                try:
                    relative_path = item.relative_to(project_dir)
                    depth = len(relative_path.parts) - 1  # Don't count filename
                    depth_counts[depth] = depth_counts.get(depth, 0) + 1
                except ValueError:
                    continue  # Skip files outside project dir
                
                # Size analysis
                try:
                    structure['total_size'] += item.stat().st_size
                    structure['total_files'] += 1
                except (OSError, PermissionError):
                    continue
        
        # Store results
        structure['directories'] = dict(sorted(dir_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        structure['file_types'] = dict(sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True)[:15])
        structure['depth_analysis'] = depth_counts
        
        # Calculate organization score (0.0 to 1.0)
        structure['organization_score'] = _calculate_organization_score(structure)
        
    except Exception as e:
        structure['error'] = str(e)
    
    return structure


def analyze_technology_stack(project_dir: Path) -> Dict[str, Any]:
    """Analyze technology stack from files and configuration"""
    
    stack = {
        'languages': {},
        'frameworks': set(),
        'databases': set(),
        'tools': set(),
        'package_managers': set(),
        'confidence_score': 0.0
    }
    
    try:
        # Language detection by file extensions
        language_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React/TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.rs': 'Rust',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'Objective-C'
        }
        
        # Count files by language
        for ext, lang in language_extensions.items():
            files = list(project_dir.glob(f'**/*{ext}'))
            if files:
                stack['languages'][lang] = len(files)
        
        # Framework detection from key files
        framework_indicators = {
            'package.json': _analyze_package_json,
            'requirements.txt': _analyze_requirements_txt,
            'Cargo.toml': _analyze_cargo_toml,
            'pom.xml': _analyze_pom_xml,
            'build.gradle': _analyze_gradle,
            'composer.json': _analyze_composer_json,
            'Gemfile': _analyze_gemfile
        }
        
        for filename, analyzer_func in framework_indicators.items():
            file_path = project_dir / filename
            if file_path.exists():
                result = analyzer_func(file_path)
                if result:
                    stack['frameworks'].update(result.get('frameworks', []))
                    stack['databases'].update(result.get('databases', []))
                    stack['tools'].update(result.get('tools', []))
                    stack['package_managers'].add(result.get('package_manager', 'unknown'))
        
        # Docker detection
        if (project_dir / 'Dockerfile').exists():
            stack['tools'].add('Docker')
        if (project_dir / 'docker-compose.yml').exists() or (project_dir / 'docker-compose.yaml').exists():
            stack['tools'].add('Docker Compose')
        
        # CI/CD detection
        if (project_dir / '.github' / 'workflows').exists():
            stack['tools'].add('GitHub Actions')
        if (project_dir / '.gitlab-ci.yml').exists():
            stack['tools'].add('GitLab CI')
        
        # Configuration management
        if (project_dir / '.env').exists():
            stack['tools'].add('Environment Variables')
        
        # Convert sets to lists for JSON serialization
        stack['frameworks'] = sorted(list(stack['frameworks']))
        stack['databases'] = sorted(list(stack['databases']))
        stack['tools'] = sorted(list(stack['tools']))
        stack['package_managers'] = sorted(list(stack['package_managers']))
        
        # Calculate confidence score
        stack['confidence_score'] = _calculate_tech_confidence(stack)
        
    except Exception as e:
        stack['error'] = str(e)
    
    return stack


def analyze_recent_activity(project_dir: Path, days_back: int = 30) -> Dict[str, Any]:
    """Analyze recent file activity and changes"""
    
    activity = {
        'recent_files': [],
        'active_directories': {},
        'file_change_patterns': {},
        'development_intensity': 0.0,
        'analysis_period_days': days_back
    }
    
    try:
        cutoff_time = time.time() - (days_back * 24 * 3600)
        
        # Find recently modified files
        recent_files = []
        dir_activity = {}
        
        for file_path in project_dir.rglob('*'):
            if not file_path.is_file():
                continue
            
            try:
                mtime = file_path.stat().st_mtime
                if mtime > cutoff_time:
                    relative_path = str(file_path.relative_to(project_dir))
                    recent_files.append({
                        'path': relative_path,
                        'modified': datetime.fromtimestamp(mtime).isoformat(),
                        'days_ago': (time.time() - mtime) / (24 * 3600)
                    })
                    
                    # Track directory activity
                    dir_name = file_path.parent.name
                    dir_activity[dir_name] = dir_activity.get(dir_name, 0) + 1
                    
            except (OSError, ValueError):
                continue
        
        # Sort by recency
        recent_files.sort(key=lambda x: x['days_ago'])
        activity['recent_files'] = recent_files[:50]  # Limit to 50 most recent
        
        # Directory activity analysis
        activity['active_directories'] = dict(sorted(dir_activity.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # File change patterns
        change_patterns = {}
        for file_info in recent_files:
            ext = Path(file_info['path']).suffix.lower() or 'no_extension'
            change_patterns[ext] = change_patterns.get(ext, 0) + 1
        
        activity['file_change_patterns'] = dict(sorted(change_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Development intensity score (files changed per day)
        if recent_files:
            activity['development_intensity'] = len(recent_files) / days_back
        
    except Exception as e:
        activity['error'] = str(e)
    
    return activity


def analyze_code_quality_indicators(project_dir: Path) -> Dict[str, Any]:
    """Analyze indicators of code quality and maintainability"""
    
    quality = {
        'has_tests': False,
        'has_documentation': False,
        'has_linting': False,
        'has_type_checking': False,
        'has_ci_cd': False,
        'quality_score': 0.0,
        'indicators': {}
    }
    
    try:
        # Test presence
        test_patterns = ['**/test_*.py', '**/tests/**/*.py', '**/*.test.js', '**/spec/**/*.js']
        for pattern in test_patterns:
            if list(project_dir.glob(pattern)):
                quality['has_tests'] = True
                break
        
        # Documentation presence
        doc_files = ['README.md', 'README.rst', 'docs/', 'documentation/']
        for doc in doc_files:
            if (project_dir / doc).exists():
                quality['has_documentation'] = True
                break
        
        # Linting configuration
        lint_configs = ['.eslintrc', '.pylintrc', 'setup.cfg', 'pyproject.toml', 'tslint.json']
        for config in lint_configs:
            if (project_dir / config).exists():
                quality['has_linting'] = True
                break
        
        # Type checking
        type_configs = ['mypy.ini', 'tsconfig.json', '.mypy.ini']
        for config in type_configs:
            if (project_dir / config).exists():
                quality['has_type_checking'] = True
                break
        
        # CI/CD presence
        ci_indicators = ['.github/workflows', '.gitlab-ci.yml', 'Jenkinsfile', '.travis.yml']
        for indicator in ci_indicators:
            if (project_dir / indicator).exists():
                quality['has_ci_cd'] = True
                break
        
        # Calculate quality score
        indicators = [
            quality['has_tests'],
            quality['has_documentation'], 
            quality['has_linting'],
            quality['has_type_checking'],
            quality['has_ci_cd']
        ]
        quality['quality_score'] = sum(indicators) / len(indicators)
        
        # Detailed indicators
        quality['indicators'] = {
            'testing_framework': quality['has_tests'],
            'documentation_system': quality['has_documentation'],
            'linting_system': quality['has_linting'],
            'type_system': quality['has_type_checking'],
            'automation_pipeline': quality['has_ci_cd']
        }
        
    except Exception as e:
        quality['error'] = str(e)
    
    return quality


def create_project_summary(project_dir: Path, max_tokens: int = 1500) -> Dict[str, Any]:
    """Create a comprehensive project summary combining all analysis functions"""
    
    summary = {
        'project_name': project_dir.name,
        'analysis_timestamp': datetime.now().isoformat(),
        'structure': analyze_project_structure(project_dir),
        'technology_stack': analyze_technology_stack(project_dir),
        'recent_activity': analyze_recent_activity(project_dir, days_back=14),  # Last 2 weeks
        'quality_indicators': analyze_code_quality_indicators(project_dir)
    }
    
    # Calculate overall project health score
    scores = []
    if 'organization_score' in summary['structure']:
        scores.append(summary['structure']['organization_score'])
    if 'confidence_score' in summary['technology_stack']:
        scores.append(summary['technology_stack']['confidence_score'])
    if 'quality_score' in summary['quality_indicators']:
        scores.append(summary['quality_indicators']['quality_score'])
    
    summary['project_health_score'] = sum(scores) / len(scores) if scores else 0.0
    
    # Token management
    summary_text = json.dumps(summary, indent=2)
    estimated_tokens = estimate_tokens(summary_text)
    
    if estimated_tokens > max_tokens:
        summary = _truncate_project_summary(summary, max_tokens)
        estimated_tokens = estimate_tokens(json.dumps(summary, indent=2))
    
    summary['meta'] = {
        'estimated_tokens': estimated_tokens,
        'analysis_version': '1.0'
    }
    
    return summary


# Helper functions for technology stack analysis

def _analyze_package_json(file_path: Path) -> Dict[str, Any]:
    """Analyze package.json for Node.js frameworks and tools"""
    try:
        content = read_file_safely(file_path, max_size=50000)
        if not content or content.startswith('['):
            return {}
        
        package_data = json.loads(content)
        
        result = {
            'package_manager': 'npm',
            'frameworks': set(),
            'databases': set(),
            'tools': set()
        }
        
        # Analyze dependencies
        all_deps = {}
        all_deps.update(package_data.get('dependencies', {}))
        all_deps.update(package_data.get('devDependencies', {}))
        
        framework_mapping = {
            'react': 'React',
            'vue': 'Vue.js', 
            'angular': 'Angular',
            'next': 'Next.js',
            'nuxt': 'Nuxt.js',
            'express': 'Express.js',
            'fastify': 'Fastify',
            'koa': 'Koa.js',
            'nest': 'NestJS'
        }
        
        database_mapping = {
            'mongodb': 'MongoDB',
            'mongoose': 'MongoDB',
            'postgres': 'PostgreSQL',
            'mysql': 'MySQL',
            'sqlite': 'SQLite',
            'redis': 'Redis'
        }
        
        tool_mapping = {
            'webpack': 'Webpack',
            'babel': 'Babel',
            'eslint': 'ESLint',
            'jest': 'Jest',
            'cypress': 'Cypress',
            'typescript': 'TypeScript'
        }
        
        for dep in all_deps.keys():
            dep_lower = dep.lower()
            
            for key, framework in framework_mapping.items():
                if key in dep_lower:
                    result['frameworks'].add(framework)
            
            for key, database in database_mapping.items():
                if key in dep_lower:
                    result['databases'].add(database)
                    
            for key, tool in tool_mapping.items():
                if key in dep_lower:
                    result['tools'].add(tool)
        
        return result
        
    except (json.JSONDecodeError, Exception):
        return {}


def _analyze_requirements_txt(file_path: Path) -> Dict[str, Any]:
    """Analyze requirements.txt for Python frameworks and tools"""
    try:
        content = read_file_safely(file_path, max_size=50000)
        if not content:
            return {}
        
        result = {
            'package_manager': 'pip',
            'frameworks': set(),
            'databases': set(),
            'tools': set()
        }
        
        # Parse requirements
        lines = content.lower().split('\n')
        requirements = []
        for line in lines:
            line = line.strip().split('==')[0].split('>=')[0].split('<=')[0]
            if line and not line.startswith('#'):
                requirements.append(line)
        
        framework_mapping = {
            'django': 'Django',
            'flask': 'Flask', 
            'fastapi': 'FastAPI',
            'tornado': 'Tornado',
            'pyramid': 'Pyramid',
            'starlette': 'Starlette'
        }
        
        database_mapping = {
            'pymongo': 'MongoDB',
            'psycopg2': 'PostgreSQL',
            'mysql-connector': 'MySQL',
            'sqlite3': 'SQLite',
            'redis': 'Redis',
            'sqlalchemy': 'SQLAlchemy'
        }
        
        tool_mapping = {
            'pytest': 'PyTest',
            'black': 'Black',
            'mypy': 'MyPy',
            'flake8': 'Flake8',
            'celery': 'Celery'
        }
        
        for req in requirements:
            for key, framework in framework_mapping.items():
                if key in req:
                    result['frameworks'].add(framework)
            
            for key, database in database_mapping.items():
                if key in req:
                    result['databases'].add(database)
                    
            for key, tool in tool_mapping.items():
                if key in req:
                    result['tools'].add(tool)
        
        return result
        
    except Exception:
        return {}


def _analyze_cargo_toml(file_path: Path) -> Dict[str, Any]:
    """Analyze Cargo.toml for Rust frameworks and tools"""
    # Simplified implementation - could be expanded
    return {
        'package_manager': 'cargo',
        'frameworks': ['Rust'],
        'databases': set(),
        'tools': set()
    }


def _analyze_pom_xml(file_path: Path) -> Dict[str, Any]:
    """Analyze pom.xml for Java frameworks and tools"""
    return {
        'package_manager': 'maven',
        'frameworks': ['Java/Maven'],
        'databases': set(),
        'tools': set()
    }


def _analyze_gradle(file_path: Path) -> Dict[str, Any]:
    """Analyze Gradle files"""
    return {
        'package_manager': 'gradle',
        'frameworks': ['Java/Gradle'],
        'databases': set(),
        'tools': set()
    }


def _analyze_composer_json(file_path: Path) -> Dict[str, Any]:
    """Analyze composer.json for PHP frameworks"""
    return {
        'package_manager': 'composer',
        'frameworks': ['PHP'],
        'databases': set(),
        'tools': set()
    }


def _analyze_gemfile(file_path: Path) -> Dict[str, Any]:
    """Analyze Gemfile for Ruby frameworks"""
    return {
        'package_manager': 'bundler',
        'frameworks': ['Ruby'],
        'databases': set(),
        'tools': set()
    }


def _calculate_organization_score(structure: Dict[str, Any]) -> float:
    """Calculate organization score based on directory structure"""
    score = 0.0
    
    # Points for good organization patterns
    directories = structure.get('directories', {})
    
    # Bonus for common good patterns
    good_patterns = ['src', 'lib', 'tests', 'test', 'docs', 'config', 'utils', 'common']
    for pattern in good_patterns:
        if pattern in directories:
            score += 0.1
    
    # Penalty for too many files in root
    root_files = directories.get('claude-l9-template', 0)  # Adjust for actual root name
    if root_files > 20:
        score -= 0.2
    elif root_files > 10:
        score -= 0.1
    
    # Bonus for reasonable depth distribution
    depth_analysis = structure.get('depth_analysis', {})
    if depth_analysis:
        avg_depth = sum(depth * count for depth, count in depth_analysis.items()) / sum(depth_analysis.values())
        if 1.5 <= avg_depth <= 3.0:  # Good depth range
            score += 0.2
    
    return min(1.0, max(0.0, score))  # Clamp to 0-1 range


def _calculate_tech_confidence(stack: Dict[str, Any]) -> float:
    """Calculate confidence score for technology stack detection"""
    score = 0.0
    
    # Points for detected components
    if stack.get('languages'):
        score += 0.3
    if stack.get('frameworks'):
        score += 0.3  
    if stack.get('tools'):
        score += 0.2
    if stack.get('databases'):
        score += 0.1
    if stack.get('package_managers'):
        score += 0.1
    
    return min(1.0, score)


def _truncate_project_summary(summary: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
    """Truncate project summary to fit token budget"""
    
    # Prioritized truncation order
    truncation_steps = [
        ('recent_activity.recent_files', 10),
        ('structure.directories', 10),  
        ('structure.file_types', 8),
        ('technology_stack.frameworks', 8),
        ('technology_stack.tools', 6),
        ('recent_activity.active_directories', 6)
    ]
    
    for field_path, limit in truncation_steps:
        if '.' in field_path:
            parent, child = field_path.split('.')
            if parent in summary and child in summary[parent]:
                if isinstance(summary[parent][child], (list, dict)):
                    if len(summary[parent][child]) > limit:
                        if isinstance(summary[parent][child], dict):
                            keys = list(summary[parent][child].keys())[:limit]
                            summary[parent][child] = {k: summary[parent][child][k] for k in keys}
                        else:
                            summary[parent][child] = summary[parent][child][:limit]
        
        # Check if we're within budget
        current_tokens = estimate_tokens(json.dumps(summary, indent=2))
        if current_tokens <= max_tokens:
            break
    
    return summary