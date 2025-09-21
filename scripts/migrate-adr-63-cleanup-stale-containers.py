#!/usr/bin/env python3
"""
ADR-63: Migration Script - Clean up stale containers with wrong mounts

This one-time cleanup removes containers that have inaccessible or wrong mount paths,
preparing the system for the mount validation fix.

Run this during deployment to clean up existing stale containers.
"""

import os
import sys
import docker
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_stale_containers(dry_run=False):
    """
    Clean up containers with inaccessible or suspicious mount paths

    Args:
        dry_run: If True, only report what would be removed without removing

    Returns:
        Number of containers removed
    """
    logger.info("="*60)
    logger.info("ADR-63 MIGRATION: Stale Container Cleanup")
    logger.info("="*60)

    if dry_run:
        logger.info("DRY RUN MODE - No containers will be removed")

    try:
        docker_client = docker.from_env()
    except Exception as e:
        logger.error(f"Failed to connect to Docker: {e}")
        return 0

    # Find all L9-managed containers
    containers = docker_client.containers.list(
        all=True,
        filters={'label': 'com.l9.managed=true'}
    )

    logger.info(f"Found {len(containers)} L9-managed containers")

    removed_count = 0
    kept_count = 0
    issues_found = []

    for container in containers:
        project = container.labels.get('com.l9.project', 'unknown')
        container_info = f"{container.name} ({container.id[:12]}) [project: {project}]"

        # Check container mounts
        try:
            mounts = container.attrs.get('Mounts', [])
            workspace_mounts = [m for m in mounts if m.get('Destination') == '/workspace']

            if not workspace_mounts:
                issue = f"No /workspace mount"
                issues_found.append((container_info, issue))

                if not dry_run:
                    logger.warning(f"Removing {container_info}: {issue}")
                    container.remove(force=True)
                    removed_count += 1
                else:
                    logger.info(f"Would remove {container_info}: {issue}")
                    removed_count += 1
                continue

            # Check if mount source exists and is accessible
            mount = workspace_mounts[0]
            source_path = mount.get('Source')

            if not source_path:
                issue = "Mount source is None"
                issues_found.append((container_info, issue))

                if not dry_run:
                    logger.warning(f"Removing {container_info}: {issue}")
                    container.remove(force=True)
                    removed_count += 1
                else:
                    logger.info(f"Would remove {container_info}: {issue}")
                    removed_count += 1
                continue

            # Check suspicious paths
            suspicious_patterns = [
                '/tmp/test',  # Generic test path
                '/tmp/neural-novelist-test',  # Known problematic test path
                '/tmp/adr-',  # ADR test paths
                '/private/tmp/test',  # macOS temp test paths
            ]

            is_suspicious = any(source_path.startswith(p) for p in suspicious_patterns)

            if is_suspicious:
                issue = f"Suspicious test path: {source_path}"
                issues_found.append((container_info, issue))

                if not dry_run:
                    logger.warning(f"Removing {container_info}: {issue}")
                    container.remove(force=True)
                    removed_count += 1
                else:
                    logger.info(f"Would remove {container_info}: {issue}")
                    removed_count += 1
                continue

            # Check if path exists
            if not os.path.exists(source_path):
                issue = f"Mount path doesn't exist: {source_path}"
                issues_found.append((container_info, issue))

                if not dry_run:
                    logger.warning(f"Removing {container_info}: {issue}")
                    container.remove(force=True)
                    removed_count += 1
                else:
                    logger.info(f"Would remove {container_info}: {issue}")
                    removed_count += 1
                continue

            # Check if container is stopped and old
            if container.status != 'running':
                created_str = container.labels.get('com.l9.created')
                if created_str:
                    try:
                        created = int(created_str)
                        age_hours = (datetime.now().timestamp() - created) / 3600

                        if age_hours > 24:  # Stopped for more than 24 hours
                            issue = f"Stopped for {age_hours:.1f} hours"
                            issues_found.append((container_info, issue))

                            if not dry_run:
                                logger.info(f"Removing old stopped {container_info}: {issue}")
                                container.remove(force=True)
                                removed_count += 1
                            else:
                                logger.info(f"Would remove {container_info}: {issue}")
                                removed_count += 1
                            continue
                    except (ValueError, TypeError):
                        pass

            # Container seems valid
            logger.debug(f"Keeping {container_info}: mount={source_path}, status={container.status}")
            kept_count += 1

        except docker.errors.NotFound:
            logger.debug(f"Container {container_info} disappeared during inspection")
            continue
        except Exception as e:
            logger.error(f"Error inspecting {container_info}: {e}")
            continue

    # Report results
    logger.info("\n" + "="*60)
    logger.info("MIGRATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total containers found: {len(containers)}")
    logger.info(f"Containers removed: {removed_count}")
    logger.info(f"Containers kept: {kept_count}")

    if issues_found:
        logger.info("\nIssues found:")
        for container_info, issue in issues_found:
            logger.info(f"  - {container_info}: {issue}")

    if dry_run:
        logger.info("\nDRY RUN COMPLETE - No changes made")
        logger.info("Run without --dry-run to actually remove containers")
    else:
        logger.info(f"\n✅ Migration complete: Removed {removed_count} stale containers")

    return removed_count


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ADR-63 Migration: Clean up stale containers with wrong mounts"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report what would be removed without actually removing'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run cleanup
    removed = cleanup_stale_containers(dry_run=args.dry_run)

    # Exit with appropriate code
    sys.exit(0 if removed == 0 or args.dry_run else 0)


if __name__ == "__main__":
    main()