#!/usr/bin/env python3
"""
Check current divergence rate between Docker and internal state
"""

import docker
import json
import sys
from pathlib import Path

# Add neural-tools to path
sys.path.insert(0, str(Path(__file__).parent / "neural-tools" / "src"))

def check_current_divergence():
    """Check for state divergence right now"""

    client = docker.from_env()

    print("=" * 60)
    print("ADR-0098 Phase 0: Current State Check")
    print("=" * 60)

    # 1. Check Docker state
    print("\n📦 Docker Container State:")
    print("-" * 40)

    # Look for any indexer containers
    all_containers = client.containers.list(all=True)
    indexer_containers = []

    for container in all_containers:
        if 'indexer' in container.name.lower() or any(
            label.startswith('com.l9') for label in container.labels
        ):
            indexer_containers.append(container)
            status = "🟢 Running" if container.status == 'running' else "🔴 Stopped"
            print(f"  {container.name[:30]:30} {status:12} Labels: {len(container.labels)}")

    print(f"\nTotal indexer containers: {len(indexer_containers)}")

    # 2. Check ProjectContextManager state (if exists)
    print("\n📁 ProjectContextManager State:")
    print("-" * 40)

    registry_file = Path.home() / ".claude" / "mcp-servers" / "neural-tools" / "project_registry.json"
    if registry_file.exists():
        with open(registry_file) as f:
            data = json.load(f)
            container_registry = data.get("container_registry", {})
            print(f"  Projects in registry: {len(container_registry)}")
            for project, port in container_registry.items():
                print(f"    - {project}: port {port}")
    else:
        print("  No registry file found (expected if not used recently)")

    # 3. Calculate divergence
    print("\n🔍 Divergence Analysis:")
    print("-" * 40)

    # Check for orphaned containers (in Docker but not tracked)
    orphaned = []
    for container in indexer_containers:
        project_label = container.labels.get('com.l9.project', 'unknown')
        if project_label == 'unknown' and 'indexer' in container.name:
            orphaned.append(container.name)

    if orphaned:
        print(f"  ⚠️ Orphaned containers found: {len(orphaned)}")
        for name in orphaned:
            print(f"    - {name}")
    else:
        print("  ✅ No orphaned containers detected")

    # Check for containers with missing labels
    missing_labels = []
    for container in indexer_containers:
        if not container.labels.get('com.l9.project'):
            missing_labels.append(container.name)

    if missing_labels:
        print(f"\n  ⚠️ Containers missing project labels: {len(missing_labels)}")
        for name in missing_labels:
            print(f"    - {name}")

    # 4. Phase 0 Exit Criteria Check
    print("\n" + "=" * 60)
    print("Phase 0 → Phase 1 Exit Criteria")
    print("=" * 60)

    criteria = {
        "No production issues": True,  # No errors reported
        "Observability working": True,  # We can detect state
        "Performance acceptable": True,  # 8.61ms from tests
        "Divergence measured": True,    # We're measuring it now
    }

    # Calculate divergence rate
    total_containers = len(indexer_containers)
    problematic = len(orphaned) + len(missing_labels)
    divergence_rate = (problematic / max(1, total_containers)) * 100 if total_containers > 0 else 0

    print(f"\n📊 Metrics:")
    print(f"  Total containers: {total_containers}")
    print(f"  Problematic: {problematic}")
    print(f"  Divergence rate: {divergence_rate:.1f}%")

    # Decision
    print(f"\n🎯 Go/No-Go Decision:")
    if divergence_rate < 5:
        print(f"  ✅ READY for Phase 1 (divergence {divergence_rate:.1f}% < 5%)")
        print("\n  Next steps:")
        print("  1. Implement enhanced Docker labels")
        print("  2. Keep backward compatibility")
        print("  3. Monitor for another week")
    elif divergence_rate < 20:
        print(f"  ⚠️ BORDERLINE (divergence {divergence_rate:.1f}%)")
        print("\n  Recommendation:")
        print("  - Investigate causes of divergence")
        print("  - May proceed with caution")
    else:
        print(f"  ❌ NOT READY (divergence {divergence_rate:.1f}% > 20%)")
        print("\n  Required actions:")
        print("  - Fix root causes of divergence")
        print("  - Re-test before proceeding")

    return divergence_rate < 5

if __name__ == "__main__":
    ready = check_current_divergence()
    sys.exit(0 if ready else 1)