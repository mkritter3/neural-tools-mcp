#!/usr/bin/env python3
"""
Centralized Collection Naming Configuration (ADR-0040)
Single source of truth for Qdrant collection naming conventions.
Enhanced with dual-write support per expert consensus.

L9 2025 Standard: September 12, 2025
MCP Protocol: 2025-06-18

Author: L9 Engineering Team
Reviewers: Gemini 2.5 Pro, Grok-4
"""

import re
import os
import logging
from typing import List

logger = logging.getLogger(__name__)


class CollectionNamingManager:
    """
    Single source of truth for Qdrant collection naming conventions.

    Implements ADR-0039 for centralized configuration, solving the
    collection naming scatter problem across MCP, indexer, and API services.

    Standard naming convention: project-{name}
    - Clean, no suffixes for default code collections
    - Sanitized to meet Qdrant requirements
    - Backward compatible during migration
    """

    # L9 Standard: Clean naming without suffixes (ADR-0039)
    # Environment override for advanced users (ADR-0037 compliance)
    _template = os.environ.get("COLLECTION_NAME_TEMPLATE", "project-{name}")

    @classmethod
    def get_collection_name(cls, project_name: str, collection_type: str = "code") -> str:
        """
        Get standardized collection name following L9 2025 standards.

        Args:
            project_name: Name of the project
            collection_type: Type of collection (code, docs, assets) - future expansion

        Returns:
            Sanitized, standardized collection name

        Example:
            >>> get_collection_name("Claude-L9-Template")
            'project-claude-l9-template'
        """
        # ADR-0037: Environment variable priority
        override = os.getenv(f"COLLECTION_NAME_OVERRIDE_{project_name.upper().replace('-', '_')}")
        if override:
            logger.info(f"Using override collection name for {project_name}: {override}")
            return override

        # Sanitize and format
        sanitized = cls._sanitize(project_name)
        collection_name = cls._template.format(name=sanitized)

        # Future: Support different collection types
        # if collection_type != "code":
        #     collection_name = f"{collection_name}-{collection_type}"

        return collection_name

    @classmethod
    def get_legacy_collection_names(cls, project_name: str) -> List[str]:
        """
        Returns all known legacy collection names for a project.
        Used for dual-write during migration (ADR-0040).

        Args:
            project_name: Name of the project

        Returns:
            List of legacy collection names (for dual-write)

        Example:
            >>> get_legacy_collection_names("claude-l9-template")
            ['project_claude-l9-template_code', 'project_claude_l9_template_code']
        """
        # Legacy formats that actually exist in our system
        sanitized = cls._sanitize(project_name)

        # The actual format we've seen: project_claude-l9-template_code
        # (underscore prefix, original hyphens preserved, underscore suffix)
        legacy_names = [
            f"project_{project_name}_code",  # Keep original hyphens
            f"project_{sanitized.replace('-', '_')}_code",  # All underscores
            f"project_{sanitized}_code",  # Sanitized with hyphens
        ]

        # De-duplicate
        return list(dict.fromkeys(legacy_names))

    @classmethod
    def get_possible_names_for_lookup(cls, project_name: str) -> List[str]:
        """
        Returns prioritized list of possible collection names for backward compatibility.
        Used during migration period to support both new and legacy formats.

        Args:
            project_name: Name of the project

        Returns:
            List of collection names to try, in priority order

        Example:
            >>> get_possible_names_for_lookup("claude-l9-template")
            ['project-claude-l9-template', 'project_claude-l9-template_code']
        """
        canonical_name = cls.get_collection_name(project_name)

        # Get legacy formats
        legacy_names = cls.get_legacy_collection_names(project_name)

        # Some very old collections might be raw project name
        legacy_raw = project_name

        # Return de-duplicated, prioritized list
        names = [canonical_name]

        for name in legacy_names + [legacy_raw]:
            if name not in names:
                names.append(name)

        logger.debug(f"Possible collection names for {project_name}: {names}")
        return names

    @classmethod
    def parse_project_name(cls, collection_name: str) -> str:
        """
        Extract project name from collection name (migration-safe).

        Args:
            collection_name: Full collection name

        Returns:
            Extracted project name

        Raises:
            ValueError: If collection name format is invalid

        Examples:
            >>> parse_project_name("project-claude-l9-template")
            'claude-l9-template'
            >>> parse_project_name("project_claude_l9_template_code")
            'claude_l9_template'
        """
        # New standard: project-{name}
        if collection_name.startswith("project-"):
            return collection_name[8:]

        # Legacy: project_{name}_code
        elif collection_name.startswith("project_") and "_code" in collection_name:
            return collection_name[8:-5]

        # Legacy: project_{name} without _code
        elif collection_name.startswith("project_"):
            return collection_name[8:]

        # Very old: raw project name (no prefix)
        elif "/" not in collection_name and "\\" not in collection_name:
            return collection_name

        raise ValueError(f"Invalid collection name format: {collection_name}")

    @classmethod
    def _sanitize(cls, name: str) -> str:
        """
        Sanitize project name to valid Qdrant collection name.
        Qdrant requires: ^[a-zA-Z0-9_-]{1,255}$

        - Lowercase for consistency
        - Replace spaces, dots with hyphens
        - Remove invalid characters
        - Trim hyphens
        - Ensure length limits

        Args:
            name: Raw project name

        Returns:
            Sanitized name

        Raises:
            ValueError: If name results in empty string

        Examples:
            >>> _sanitize("Claude L9 Template!")
            'claude-l9-template'
            >>> _sanitize("my.project.2025")
            'my-project-2025'
        """
        # Lowercase for consistency
        name = name.lower()

        # Replace whitespace, dots, underscores with hyphens
        name = re.sub(r'[\s_.]+', '-', name)

        # Remove any character that's not alphanumeric or hyphen
        name = re.sub(r'[^a-z0-9-]', '', name)

        # Remove leading/trailing hyphens
        name = name.strip('-')

        # Collapse multiple consecutive hyphens
        name = re.sub(r'-+', '-', name)

        # Validate result
        if not name:
            raise ValueError("Project name resulted in empty sanitized name")

        # Ensure length limit (Qdrant max is 255)
        if len(name) > 255:
            name = name[:255].rstrip('-')
            logger.warning(f"Truncated collection name to 255 characters: {name}")

        return name

    @classmethod
    def validate_migration_needed(cls, collection_name: str) -> bool:
        """
        Check if a collection needs migration to new naming standard.

        Args:
            collection_name: Current collection name

        Returns:
            True if migration needed, False if already standardized

        Example:
            >>> validate_migration_needed("project_claude_l9_template_code")
            True
            >>> validate_migration_needed("project-claude-l9-template")
            False
        """
        try:
            project_name = cls.parse_project_name(collection_name)
            canonical_name = cls.get_collection_name(project_name)
            return collection_name != canonical_name
        except ValueError:
            # Unknown format, probably needs investigation
            logger.warning(f"Unknown collection format, may need manual migration: {collection_name}")
            return False


# Singleton instance for easy import
collection_naming = CollectionNamingManager()

# Export the main functions for convenience
get_collection_name = collection_naming.get_collection_name
get_legacy_collection_names = collection_naming.get_legacy_collection_names
get_possible_names_for_lookup = collection_naming.get_possible_names_for_lookup
parse_project_name = collection_naming.parse_project_name
validate_migration_needed = collection_naming.validate_migration_needed