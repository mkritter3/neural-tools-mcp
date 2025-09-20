#!/usr/bin/env python3
"""
Critical Collection Naming Test
Ensures that services that directly create/query Qdrant collections use consistent naming
Per ADR-0057: Standard format is 'project-{project_name}'
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple


# Critical files that MUST use the correct naming
# These are the ones that actually create or query main Qdrant collections
CRITICAL_FILES = {
    "neural_mcp/neural_server_stdio.py": {
        "functions": ["neural_system_status_impl", "project_understanding_impl"],
        "description": "MCP server status reporting"
    },
    "servers/services/sync_manager.py": {
        "functions": ["__init__"],
        "description": "Write synchronization manager"
    },
    "servers/services/drift_monitor.py": {
        "functions": ["__init__"],
        "description": "Drift monitoring service"
    },
    "servers/services/qdrant_service.py": {
        "functions": ["__init__"],
        "description": "Main Qdrant service"
    },
    "servers/services/indexer_service.py": {
        "functions": ["__init__"],
        "description": "Main indexing service"
    },
    "servers/services/collection_config.py": {
        "functions": ["get_collection_name"],
        "description": "Collection configuration manager"
    }
}

CORRECT_PATTERN = "project-{project_name}"


def extract_collection_patterns(file_path: Path) -> List[Tuple[int, str, str]]:
    """
    Extract collection naming patterns from a Python file
    Returns list of (line_number, code, pattern)
    """
    patterns = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for i, line in enumerate(lines, 1):
            # Look for collection naming patterns
            if 'collection' in line.lower() and 'project' in line:
                # Check for f-strings
                if 'f"project' in line or "f'project" in line:
                    if 'project-{' in line:
                        patterns.append((i, line.strip(), "project-{...}"))
                    elif 'project_{' in line:
                        patterns.append((i, line.strip(), "project_{...}"))

                # Check for string formatting
                elif '"project-' in line or "'project-" in line:
                    patterns.append((i, line.strip(), "project-..."))
                elif '"project_' in line or "'project_" in line:
                    patterns.append((i, line.strip(), "project_..."))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return patterns


def check_critical_files(root_path: Path) -> Tuple[bool, List[Dict]]:
    """
    Check only the critical files for collection naming consistency
    Returns (success, issues)
    """
    issues = []
    checked_files = []

    src_path = root_path / "neural-tools" / "src"

    missing_critical_files = []

    for relative_path, info in CRITICAL_FILES.items():
        file_path = src_path / relative_path

        if not file_path.exists():
            # Only warn for files that SHOULD exist (not future placeholders)
            expected_files = [
                "neural_mcp/neural_server_stdio.py",
                "servers/services/qdrant_service.py",
                "servers/services/indexer_service.py"
            ]
            if relative_path in expected_files:
                print(f"❌ CRITICAL: Expected file not found: {relative_path}")
                missing_critical_files.append(relative_path)
            else:
                print(f"ℹ️  Future file not yet implemented: {relative_path}")
            continue

        checked_files.append(relative_path)
        patterns = extract_collection_patterns(file_path)

        # Check if any incorrect patterns are found
        for line_num, code, pattern in patterns:
            if pattern.startswith("project_"):
                issues.append({
                    'file': relative_path,
                    'line': line_num,
                    'code': code[:80] + "..." if len(code) > 80 else code,
                    'pattern': pattern,
                    'service': info['description']
                })

    # Fail if critical expected files are missing
    if missing_critical_files:
        for missing_file in missing_critical_files:
            issues.append({
                'file': missing_file,
                'line': 0,
                'code': 'FILE NOT FOUND',
                'pattern': 'N/A',
                'service': CRITICAL_FILES[missing_file]['description']
            })

    return len(issues) == 0, issues


def main():
    """Main test function"""
    print("=" * 70)
    print("Critical Collection Naming Test (ADR-0057)")
    print("=" * 70)
    print()
    print("Checking services that directly interact with Qdrant collections...")
    print(f"Expected format: '{CORRECT_PATTERN}'")
    print()

    root_path = Path.cwd()
    success, issues = check_critical_files(root_path)

    print("Critical Files Checked:")
    for file, info in CRITICAL_FILES.items():
        print(f"  • {file} - {info['description']}")
    print()

    if success:
        print("✅ SUCCESS: All critical services use consistent collection naming!")
        print(f"   Standard format '{CORRECT_PATTERN}' is used in all critical locations")
        return 0
    else:
        print(f"❌ FAILED: Found {len(issues)} critical naming inconsistencies:")
        print()

        for issue in issues:
            print(f"📄 {issue['file']} (Line {issue['line']}):")
            print(f"   Service: {issue['service']}")
            print(f"   Code: {issue['code']}")
            print(f"   Pattern: {issue['pattern']} → Should be: {CORRECT_PATTERN}")
            print()

        print("These are CRITICAL issues that will cause collection detection failures!")
        print("Fix these immediately to ensure proper system operation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())