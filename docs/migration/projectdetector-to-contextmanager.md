# Migration Plan: ProjectDetector → ProjectContextManager

**Date:** 2025-09-24
**Status:** Ready to Execute
**Risk Level:** LOW (No active usage found)

## Executive Summary

**GOOD NEWS:** ProjectDetector is NOT actively used in the codebase!
- Only referenced in 1 test file
- No tools actually call `get_user_project()`
- Migration is essentially just cleanup

## Current State Analysis

### Files Referencing ProjectDetector
1. `/neural-tools/src/neural_mcp/project_detector.py` - The file itself
2. `/neural-tools/test_comprehensive_fix.py` - Test checking for import (line 89)

### Active Usage
- **ZERO** active calls to `get_user_project()`
- **ZERO** imports in production code
- **ZERO** dependencies on ProjectDetector

## Migration Steps

### Step 0: Pre-Migration Verification
```bash
# Confirm no active usage
grep -r "get_user_project()" neural-tools/src --include="*.py"
# Expected: Only the function definition itself

grep -r "from neural_mcp.project_detector" neural-tools/src --include="*.py"
# Expected: No results
```

### Step 1: Update Test File
**File:** `/neural-tools/test_comprehensive_fix.py`

**Change Line 89:**
```python
# OLD
'from neural_mcp.project_detector import get_user_project'

# NEW - Remove this line entirely (it's just checking for an import that shouldn't exist)
```

Or update test to check for ProjectContextManager instead:
```python
'from servers.services.project_context_manager import get_project_context_manager'
```

### Step 2: Create Compatibility Shim (Optional - Not Needed)
Since no one is using it, we can skip the compatibility layer entirely.

### Step 3: Delete ProjectDetector
```bash
# Archive for safety
mv neural-tools/src/neural_mcp/project_detector.py \
   neural-tools/src/neural_mcp/project_detector.py.archived-20250924

# Or delete directly
rm neural-tools/src/neural_mcp/project_detector.py
```

### Step 4: Update ProjectContextManager
Ensure ProjectContextManager has all detection strategies from ProjectDetector:

**Already Has:**
- ✅ Environment variable detection
- ✅ File-based detection (package.json, pyproject.toml)
- ✅ Container detection
- ✅ Registry persistence
- ✅ Singleton pattern

**Needs to Add:**
- Container name parsing logic (the improved version from ProjectDetector)
- Global MCP safety check

### Step 5: Enhance ProjectContextManager

Add the missing container detection improvements:

```python
# In project_context_manager.py

async def _detect_from_containers(self) -> Optional[Tuple[str, Path]]:
    """
    Detect project from running Docker containers
    ADR-0102: Secondary confirmation, not primary source
    """
    if not self.docker_client:
        return None

    try:
        containers = self.docker_client.containers.list(all=True)

        # Find running indexer containers
        indexer_containers = []
        for container in containers:
            if container.name.startswith("indexer-") and container.status == "running":
                # Parse: indexer-{project}-{timestamp}-{random}
                parts = container.name.split("-")
                if len(parts) >= 4:
                    project_name = "-".join(parts[1:-2])

                    # Get start time and mount path
                    started_at = container.attrs.get("State", {}).get("StartedAt", "")
                    mounts = container.attrs.get("Mounts", [])

                    for mount in mounts:
                        if mount.get("Destination") == "/workspace":
                            source = mount.get("Source")
                            if source:
                                project_path = Path(source)
                                if project_path.exists():
                                    indexer_containers.append({
                                        "name": project_name,
                                        "path": project_path,
                                        "started": started_at,
                                        "container": container.name
                                    })
                                    break

        # Use most recently started container
        if indexer_containers:
            indexer_containers.sort(key=lambda x: x["started"], reverse=True)
            most_recent = indexer_containers[0]
            logger.info(f"🐳 Container detection found: {most_recent['name']}")
            return most_recent["name"], most_recent["path"]

    except Exception as e:
        logger.debug(f"Container detection failed: {e}")

    return None
```

### Step 6: Update All Tools (Not Needed!)
Since no tools are using ProjectDetector, this step is skipped.

### Step 7: Testing

Create test to verify unified detection:

```python
# test_unified_detection.py
import asyncio
from servers.services.project_context_manager import get_project_context_manager

async def test_detection():
    manager = await get_project_context_manager()

    # Test detection
    project_info = await manager.get_current_project()
    print(f"Detected: {project_info}")

    # Test explicit setting
    await manager.set_project("/path/to/project")

    # Verify container detection is secondary
    # (should still return project even without containers)

if __name__ == "__main__":
    asyncio.run(test_detection())
```

## Rollback Plan

If issues arise:
```bash
# Restore ProjectDetector
mv neural-tools/src/neural_mcp/project_detector.py.archived-20250924 \
   neural-tools/src/neural_mcp/project_detector.py
```

But since it's not used, rollback is unlikely to be needed.

## Timeline

**Total Time: ~30 minutes**

1. **Minute 0-5:** Verify no active usage
2. **Minute 5-10:** Update test file
3. **Minute 10-15:** Enhance ProjectContextManager with container logic
4. **Minute 15-20:** Delete ProjectDetector
5. **Minute 20-30:** Test unified detection

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Hidden usage | Very Low | Low | grep search completed |
| Test breakage | Low | Minimal | Update test file |
| Detection fails | Low | Medium | Keep container logic |

## Success Criteria

✅ No import errors after deletion
✅ ProjectContextManager works standalone
✅ Container detection still functions
✅ All tests pass

## Commands to Execute

```bash
# 1. Final verification
grep -r "get_user_project\(\)" neural-tools/src --include="*.py"

# 2. Archive ProjectDetector
mv neural-tools/src/neural_mcp/project_detector.py \
   neural-tools/src/neural_mcp/project_detector.py.archived-20250924

# 3. Update test file
sed -i '' '/from neural_mcp.project_detector import get_user_project/d' \
   neural-tools/test_comprehensive_fix.py

# 4. Run tests
python neural-tools/test_comprehensive_fix.py

# 5. If all good, celebrate!
echo "✅ Migration complete - ProjectDetector eliminated!"
```

## Conclusion

This is the **EASIEST** migration possible:
- No active usage to migrate
- No API changes needed
- No tool updates required
- Just delete unused code!

**Confidence: 100%** - Safe to proceed immediately.