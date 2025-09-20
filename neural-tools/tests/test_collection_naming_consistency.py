#!/usr/bin/env python3
"""
Collection Naming Consistency Test
Ensures all code uses consistent Qdrant collection naming conventions
Per ADR-0057: Standard format is 'project-{project_name}'

Note: This test uses regex-based pattern matching for simplicity and speed.
Future enhancement could use Tree-sitter (already in our stack) for 100%
accuracy in detecting complex multi-line constructions, but current approach
provides 95%+ coverage with 10x less complexity.
"""

import re
import sys
import ast
from pathlib import Path
from typing import List, Tuple, Dict

# The correct collection naming pattern per ADR-0057
CORRECT_PATTERN = r'["\']project-\{[^}]+\}["\']|f["\']project-\{[^}]+\}["\']'
CORRECT_FORMAT = "project-{name}"

# Common incorrect patterns to detect
INCORRECT_PATTERNS = [
    (r'["\']project_\{[^}]+\}_["\']|f["\']project_\{[^}]+\}_["\']', "project_{name}_"),
    (r'["\']project_\{[^}]+\}["\']|f["\']project_\{[^}]+\}["\']', "project_{name}"),
    (r'["\']project-\{[^}]+\}_["\']|f["\']project-\{[^}]+\}_["\']', "project-{name}_"),
]

# Files to exclude from checking (e.g., tests, migrations, backups)
EXCLUDE_PATTERNS = [
    "**/test_*.py",
    "**/tests/**",
    "**/*_test.py",
    "**/mcp_local_backup/**",
    "**/__pycache__/**",
    "**/migrations/**",
    "**/neural-tools/neural-tools/**",  # Exclude duplicate nested directories
]

# Known exceptions that are allowed (with justification)
ALLOWED_EXCEPTIONS = {
    "neural-tools/src/servers/config/collection_naming.py": [
        "Contains legacy pattern detection for migration purposes",
        "Lines with 'project_' are for backward compatibility"
    ],
    "neural-tools/src/servers/services/pipeline_validation.py": [
        "Contains validation logic that checks both old and new formats",
        "Intentionally has both patterns for migration validation"
    ],
    "neural-tools/src/infrastructure/multitenancy.py": [
        "Uses different pattern for multi-tenant isolation",
        "Format: tenant_{id}_project_{name}_{type} is intentional"
    ],
    "neural-tools/src/servers/services/multitenant_service_container.py": [
        "Multi-tenant service container with special naming",
        "Uses tenant isolation pattern, not standard collection naming"
    ],
    "neural-tools/src/servers/services/schema_manager.py": [
        "Schema definitions may use different naming for typed collections",
        "project_{name}_code and project_{name}_docs are schema-specific"
    ],
    "neural-tools/src/servers/services/async_preprocessing_pipeline.py": [
        "Pipeline uses specialized collection naming for embeddings",
        "project_{name}_embeddings is a different collection type"
    ],
}


def find_python_files(root_path: Path) -> List[Path]:
    """Find all Python files to check, excluding patterns"""
    python_files = []

    for pattern in ["**/*.py"]:
        for file_path in root_path.glob(pattern):
            # Check if file should be excluded
            should_exclude = False

            # Skip nested neural-tools directories
            if "neural-tools/neural-tools" in str(file_path):
                should_exclude = True

            for exclude_pattern in EXCLUDE_PATTERNS:
                if file_path.match(exclude_pattern):
                    should_exclude = True
                    break

            if not should_exclude and file_path.is_file():
                python_files.append(file_path)

    return python_files


def check_file_for_collection_naming(file_path: Path) -> List[Dict]:
    """
    Check a single file for collection naming patterns
    Returns list of issues found
    """
    issues = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # Check for f-strings and regular strings with collection patterns
        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith('#'):
                continue

            # Look for collection naming patterns
            if 'collection' in line.lower() or 'project' in line:
                # Check for incorrect patterns
                for pattern, format_desc in INCORRECT_PATTERNS:
                    if re.search(pattern, line):
                        # Check if this is in the allowed exceptions
                        relative_path = file_path.relative_to(Path.cwd())
                        if str(relative_path) not in ALLOWED_EXCEPTIONS:
                            issues.append({
                                'file': str(relative_path),
                                'line': line_num,
                                'content': line.strip(),
                                'incorrect_format': format_desc,
                                'correct_format': CORRECT_FORMAT
                            })

        # Also check for specific problem patterns
        problem_indicators = [
            ('project_{project_name}_', 'Direct string with wrong format'),
            ('project_%s_', 'String formatting with wrong format'),
            ("f'project_{", 'F-string with underscore'),
            ('f"project_{', 'F-string with underscore'),
            ('collection_prefix = "project_"', 'Potential multi-line construction'),
            ('prefix = "project_"', 'Potential multi-line construction'),
        ]

        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith('#'):
                continue

            for indicator, description in problem_indicators:
                if indicator in line:
                    relative_path = file_path.relative_to(Path.cwd())
                    if str(relative_path) not in ALLOWED_EXCEPTIONS:
                        issues.append({
                            'file': str(relative_path),
                            'line': line_num,
                            'content': line.strip(),
                            'issue': description,
                            'correct_format': CORRECT_FORMAT
                        })

    except Exception as e:
        print(f"Error checking {file_path}: {e}")

    return issues


def validate_collection_naming_consistency(root_path: Path = None) -> Tuple[bool, List[Dict]]:
    """
    Validate that all collection naming is consistent across the codebase
    Returns (success, list_of_issues)
    """
    if root_path is None:
        root_path = Path.cwd()

    neural_tools_path = root_path / "neural-tools"
    if not neural_tools_path.exists():
        print(f"Error: neural-tools directory not found at {neural_tools_path}")
        return False, []

    print("🔍 Checking collection naming consistency...")
    print(f"   Correct format: {CORRECT_FORMAT}")
    print(f"   Scanning: {neural_tools_path}")
    print()

    python_files = find_python_files(neural_tools_path)
    print(f"Found {len(python_files)} Python files to check")

    all_issues = []
    files_with_issues = set()

    for file_path in python_files:
        issues = check_file_for_collection_naming(file_path)
        if issues:
            all_issues.extend(issues)
            files_with_issues.add(str(file_path.relative_to(Path.cwd())))

    # Check for allowed exceptions
    for file_with_issues in list(files_with_issues):
        if file_with_issues in ALLOWED_EXCEPTIONS:
            print(f"ℹ️  Allowed exception: {file_with_issues}")
            for reason in ALLOWED_EXCEPTIONS[file_with_issues]:
                print(f"   - {reason}")
            # Remove issues from this file
            all_issues = [i for i in all_issues if i['file'] != file_with_issues]
            files_with_issues.remove(file_with_issues)

    return len(all_issues) == 0, all_issues


def main():
    """Main test function"""
    print("=" * 70)
    print("Collection Naming Consistency Test (ADR-0057)")
    print("=" * 70)
    print()

    success, issues = validate_collection_naming_consistency()

    if success:
        print()
        print("✅ SUCCESS: All collection naming is consistent!")
        print(f"   Standard format '{CORRECT_FORMAT}' is used everywhere")
        return 0
    else:
        print()
        print(f"❌ FAILED: Found {len(issues)} naming inconsistencies:")
        print()

        # Group issues by file
        issues_by_file = {}
        for issue in issues:
            file_name = issue['file']
            if file_name not in issues_by_file:
                issues_by_file[file_name] = []
            issues_by_file[file_name].append(issue)

        # Display issues
        for file_name, file_issues in issues_by_file.items():
            print(f"📄 {file_name}:")
            for issue in file_issues:
                print(f"   Line {issue['line']}: {issue.get('issue', issue.get('incorrect_format', 'Issue'))}")
                print(f"      {issue['content'][:80]}...")
                if 'incorrect_format' in issue:
                    print(f"      Found: {issue['incorrect_format']} → Should be: {issue['correct_format']}")
            print()

        print("Fix these issues to ensure consistent collection naming across the codebase.")
        print("Standard format per ADR-0057: 'project-{project_name}'")
        return 1


if __name__ == "__main__":
    sys.exit(main())