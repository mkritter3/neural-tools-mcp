# ADR-0063: Container Mount Validation Regression and Test Blind Spots

**Status**: Accepted
**Date**: September 21, 2025
**Author**: L9 Engineering Team
**Severity**: Critical (User Impact) / High (System Impact)
**Confidence**: 95% - Code analysis confirms all stated issues

## Executive Summary

ADR-60 introduced a container reuse optimization that fails to verify volume mounts match the requested project path. This creates a critical failure mode where indexers run against wrong directories, appearing to work while silently failing to index the correct codebase. The regression passed all tests due to systematic blind spots in our validation framework.

**Evidence**: Line 281-296 of `indexer_orchestrator.py` reuses containers without mount verification.
**Impact**: Production indexers fail silently, indexing test directories instead of user projects.
**Scope**: Affects users who switch between projects or have stale containers from previous runs.

## The Regression

### What Broke

ADR-60 changed container management from "remove and recreate" (ADR-48) to "discover and reuse" without verifying mount correctness:

```python
# ADR-60 BROKEN CODE (current)
existing_container = await self._discover_existing_container(project_name)
if existing_container:
    # REUSES without checking mount path!
    return existing_container['id']

# ADR-48 WORKING CODE (removed)
existing_containers = self.docker_client.containers.list(
    filters={'name': f'indexer-{project_name}'}
)
for container in existing_containers:
    container.remove(force=True)  # Always fresh mount
```

### Production Evidence

```bash
# User requests indexing for neural-novelist project
Project Path: /Users/mkr/local-coding/Systems/neural-novelist

# Container found and reused (ADR-60 logic)
Container: indexer-neural-novelist-1758444657-1e5f

# But container has WRONG mount from earlier test
Actual Mount: /tmp/neural-novelist-test -> /workspace
Should Be: /Users/mkr/local-coding/Systems/neural-novelist -> /workspace

# Result: Only README.md indexed (the only file in /tmp/neural-novelist-test)
```

## Why Tests Didn't Catch This

### The Fundamental Blind Spot

**Every single test uses the same path**:

```python
# Test 1: Container conflicts
self.orchestrator.ensure_indexer('test-conflict', '/tmp/test')

# Test 2: Concurrent requests
self.orchestrator.ensure_indexer('test-concurrent', '/tmp/test')

# Test 3: Performance
self.orchestrator.ensure_indexer('test-perf', '/tmp/test')

# Test 5: Redis locking
self.orchestrator.ensure_indexer('test-redis-lock', '/tmp/test')
```

**Result**: Container reuse ALWAYS works because the mount never needs to change!

### Test Philosophy Problem

Our tests validate **MECHANISM** not **OUTCOME**:
- ✅ Does Redis locking work? (Yes)
- ✅ Do unique names prevent 409 conflicts? (Yes)
- ✅ Can we discover containers by labels? (Yes)
- ❌ Can users actually index their projects? (NO!)

We test that code executes as written, not that it solves user problems.

## Critical Test Gaps Identified

### 1. **Path Mutation Testing** [CRITICAL]
Never test container reuse with different mount paths:
```python
# MISSING TEST - Path change scenario
container_id = ensure_indexer('myproject', '/path/A')
container_id2 = ensure_indexer('myproject', '/path/B')  # Should recreate!
assert container_id != container_id2  # Currently fails!
```

### 2. **Mount Verification** [HIGH]
Never verify container mounts match requested path:
```python
# MISSING TEST - Mount validation
container = ensure_indexer('project', '/expected/path')
mounts = docker.inspect(container)['Mounts']
assert mounts[0]['Source'] == '/expected/path'  # Never checked!
```

### 3. **Stale Container Simulation** [HIGH]
Never test with pre-existing containers from old runs:
```python
# MISSING TEST - Stale container handling
# Pre-create container with wrong mount
old_container = create_container('project', mount='/wrong/path')
# Should detect wrong mount and recreate
new_container = ensure_indexer('project', '/correct/path')
assert old_container.id != new_container.id
```

### 4. **User Journey Testing** [MEDIUM]
Never simulate real usage patterns:
```python
# MISSING TEST - Project switching journey
# User works on Project A
ensure_indexer('projectA', '/home/user/projectA')
# User switches to Project B
ensure_indexer('projectB', '/home/user/projectB')
# User returns to Project A with different checkout
ensure_indexer('projectA', '/tmp/projectA-feature-branch')  # Breaks!
```

## Complete Solution Design

### The L9-Standard Fix: Restore ALL Broken ADR Goals

We need a comprehensive fix that:
1. Restores ADR-48's idempotent behavior
2. Uses ADR-44's ContainerDiscoveryService properly
3. Validates ADR-37's configuration priority
4. Maintains ADR-30's multi-container orchestration
5. Ensures ADR-52's auto-initialization works

### Implementation: Smart Container Management

```python
async def _ensure_indexer_internal(self, project_name: str, project_path: str) -> str:
    """
    L9 Standard: Verify ALL preconditions before reuse
    """
    # 1. Check for existing container
    existing = await self._discover_existing_container(project_name)

    if existing:
        # 2. VERIFY it matches our requirements
        container = self.docker_client.containers.get(existing['id'])

        # Check mount path
        mounts = container.attrs.get('Mounts', [])
        mount_matches = any(
            m['Source'] == os.path.abspath(project_path) and
            m['Destination'] == '/workspace'
            for m in mounts
        )

        # Check environment variables match current config
        env_vars = container.attrs['Config']['Env']
        env_dict = dict(e.split('=', 1) for e in env_vars)
        config_matches = (
            env_dict.get('PROJECT_NAME') == project_name and
            env_dict.get('NEO4J_PASSWORD') == os.getenv('NEO4J_PASSWORD', 'graphrag-password')
        )

        # 3. Reuse ONLY if everything matches
        if mount_matches and config_matches:
            logger.info(f"[ADR-63] Reusing container with verified configuration")
            return existing['id']
        else:
            # 4. Remove container with wrong configuration
            logger.info(f"[ADR-63] Removing container with outdated configuration")
            logger.info(f"  Mount matches: {mount_matches}, Config matches: {config_matches}")
            container.remove(force=True)
            await self._invalidate_cache(project_name)

    # 5. Use ContainerDiscoveryService if available (ADR-44)
    if self.discovery_service and self.context_manager:
        logger.info(f"[ADR-44] Using discovery service for container creation")
        container_info = await self.discovery_service.get_or_create_container(
            project_name,
            self.context_manager
        )
        return container_info['container_id']

    # 6. Fallback to direct creation
    return await self._create_container(project_name, container_name, project_path)
```

### Alternative: Pure Idempotent Approach (Simpler)

If verification is too complex, restore ADR-48's original behavior:

```python
async def _ensure_indexer_internal(self, project_name: str, project_path: str) -> str:
    """
    ADR-48 Restored: Always remove and recreate for guaranteed correctness
    """
    # Remove ANY existing container for this project
    existing = await self._discover_existing_container(project_name)
    if existing:
        container = self.docker_client.containers.get(existing['id'])
        logger.info(f"[ADR-48] Removing existing container for idempotent recreation")
        container.remove(force=True)
        await self._invalidate_cache(project_name)

    # Always create fresh with correct configuration
    return await self._create_container(project_name, project_path)
```

## Solution Design

### Immediate Fix (Stop the Bleeding)

**Option A: Restore ADR-48 Behavior** (Recommended)
```python
async def _ensure_indexer_internal(self, project_name: str, project_path: str):
    # ALWAYS remove existing containers before creating new ones
    existing = self._discover_existing_container(project_name)
    if existing:
        logger.info(f"[ADR-63] Removing existing container for fresh mount")
        container = self.docker_client.containers.get(existing['id'])
        container.remove(force=True)

    # Create with correct mount
    return await self._create_container(project_name, project_path)
```

**Option B: Verify Mount Before Reuse**
```python
async def _ensure_indexer_internal(self, project_name: str, project_path: str):
    existing = self._discover_existing_container(project_name)
    if existing:
        # Verify mount matches
        container = self.docker_client.containers.get(existing['id'])
        mounts = container.attrs['Mounts']
        if mounts and mounts[0]['Source'] == project_path:
            return existing['id']  # Reuse only if mount matches
        else:
            logger.info(f"[ADR-63] Mount mismatch, removing container")
            container.remove(force=True)

    return await self._create_container(project_name, project_path)
```

### Enhanced Validation Framework

#### 1. Path Mutation Test Suite
```python
async def test_path_change_forces_recreation(self):
    """Containers must be recreated when path changes"""
    paths = [
        '/tmp/test-v1',
        '/home/user/projects/test',
        '/tmp/test-v2'
    ]

    container_ids = []
    for path in paths:
        container_id = await self.orchestrator.ensure_indexer('test-project', path)
        container_ids.append(container_id)

        # Verify mount is correct
        container = self.docker_client.containers.get(container_id)
        mounts = container.attrs['Mounts']
        assert mounts[0]['Source'] == path, f"Wrong mount: {mounts[0]['Source']} != {path}"

    # All should be different containers
    assert len(set(container_ids)) == len(paths), "Container reused with wrong mount!"
```

#### 2. Stale Container Pollution Test
```python
async def test_stale_container_with_wrong_mount(self):
    """Pre-existing containers with wrong mounts must be replaced"""

    # Pollute: Create container with wrong mount
    self.docker_client.containers.run(
        image='l9-neural-indexer:production',
        name='indexer-polluted-1234567-abcd',
        labels={'com.l9.project': 'polluted-test'},
        volumes={'/wrong/path': {'bind': '/workspace', 'mode': 'ro'}},
        detach=True
    )

    # Ensure indexer should detect wrong mount and recreate
    correct_path = '/correct/path'
    os.makedirs(correct_path, exist_ok=True)

    container_id = await self.orchestrator.ensure_indexer('polluted-test', correct_path)

    # Verify new container has correct mount
    container = self.docker_client.containers.get(container_id)
    mounts = container.attrs['Mounts']
    assert mounts[0]['Source'] == correct_path
```

#### 3. User Journey Simulation
```python
async def test_realistic_user_journey(self):
    """Simulate real user switching between projects"""

    # Journey: Developer's Monday morning
    journeys = [
        ('morning-standup', '/Users/dev/work/main-app'),
        ('urgent-bugfix', '/tmp/hotfix-branch'),
        ('morning-standup', '/Users/dev/work/main-app'),  # Return to main
        ('new-feature', '/Users/dev/work/feature-xyz'),
        ('code-review', '/tmp/review-pr-123'),
        ('morning-standup', '/Users/dev/work/main-app-v2')  # Different checkout
    ]

    for project, path in journeys:
        os.makedirs(path, exist_ok=True)

        # Create dummy files to index
        Path(f"{path}/app.py").write_text("print('hello')")

        container_id = await self.orchestrator.ensure_indexer(project, path)

        # Verify indexer can actually see the files
        response = requests.get(f"http://localhost:{port}/status")
        assert response.json()['files_visible'] > 0, f"Indexer can't see files in {path}"
```

#### 4. Pragmatic Test Priority (Start Simple)

Per Grok's feedback, start with 2-3 focused tests before implementing chaos testing:

```python
# Priority 1: The test that would have caught this bug
async def test_mount_changes_force_recreation(self):
    """MUST PASS: Container recreation on path change"""
    container1 = await ensure_indexer('project', '/path/A')
    container2 = await ensure_indexer('project', '/path/B')
    assert container1 != container2

# Priority 2: Stale container with wrong mount
async def test_stale_container_wrong_mount(self):
    """MUST PASS: Replace containers with wrong mounts"""
    # Pre-create with wrong mount
    create_container('project', mount='/wrong')
    # Should detect and replace
    new_container = await ensure_indexer('project', '/correct')
    # Verify mount is correct
    assert get_mount(new_container) == '/correct'

# Priority 3: Config change forces recreation
async def test_env_var_change_forces_recreation(self):
    """MUST PASS: Environment variable changes trigger recreation"""
    os.environ['NEO4J_PASSWORD'] = 'password1'
    container1 = await ensure_indexer('project', '/path')
    os.environ['NEO4J_PASSWORD'] = 'password2'
    container2 = await ensure_indexer('project', '/path')
    assert container1 != container2
```

### 5. Chaos Testing (After Basics Pass)
```python
async def test_chaos_random_container_states(self):
    """Only after the 3 priority tests pass reliably"""

    for i in range(10):
        # Randomly create containers with different states
        if random.choice([True, False]):
            # Create stale container
            self.docker_client.containers.run(
                image='l9-neural-indexer:production',
                name=f'indexer-chaos-{i}',
                labels={'com.l9.project': f'chaos-{i}'},
                volumes={f'/random/path/{i}': {'bind': '/workspace', 'mode': 'ro'}},
                detach=True
            )

        # Now try to use it correctly
        correct_path = f'/correct/path/{i}'
        os.makedirs(correct_path, exist_ok=True)

        container_id = await self.orchestrator.ensure_indexer(f'chaos-{i}', correct_path)

        # Must always have correct mount regardless of pre-existing state
        container = self.docker_client.containers.get(container_id)
        mounts = container.attrs['Mounts']
        assert mounts[0]['Source'] == correct_path
```

## Deployment Validation Requirements

### Pre-Deployment Gates

1. **Path Mutation Coverage**: ≥80% of tests must use different paths
2. **Mount Verification**: 100% of container creations must verify mounts
3. **Stale Container Tests**: At least 3 pollution scenarios must pass
4. **Journey Tests**: At least 2 realistic user journeys must complete

### CI/CD Integration

```yaml
# .github/workflows/validate-deployment.yml
deployment-validation:
  steps:
    - name: Path Mutation Tests
      run: pytest tests/test_path_mutations.py -v

    - name: Stale Container Tests
      run: |
        # Pre-pollute environment
        docker run -d --name indexer-stale --label com.l9.project=test \
          -v /wrong:/workspace l9-neural-indexer:production
        pytest tests/test_stale_containers.py -v

    - name: User Journey Tests
      run: pytest tests/test_user_journeys.py --slow

    - name: Chaos Testing
      run: pytest tests/test_chaos.py --iterations=50
```

## L9 Engineering Assessment

### Verified Issues (Code-Confirmed)

1. **Mount Validation Missing** ✅ CONFIRMED
   - Line 281-296: `_ensure_indexer_internal` returns existing container without checking mounts
   - Line 289: Updates tracking with NEW path while keeping OLD container
   - Zero mount verification logic exists

2. **ContainerDiscoveryService Orphaned** ✅ CONFIRMED
   - Line 113: Service initialized
   - Grep results: No calls to `self.discovery_service` methods anywhere
   - Dead code violating ADR-44's architecture

3. **Test Path Homogeneity** ✅ CONFIRMED
   - All tests use `/tmp/test` (lines 158, 203, 347, 364)
   - Zero path variation testing
   - Container reuse works perfectly when paths never change

4. **ADR-48 Logic Removed** ✅ CONFIRMED
   - No container removal before creation
   - Only removal is 7-day garbage collection (line 452)
   - Idempotency guarantee completely lost

### Performance vs Correctness Trade-off

ADR-60 optimized for:
- ✅ Faster container startup (reuse existing)
- ✅ Reduced Docker API calls
- ✅ Lower resource churn

But sacrificed:
- ❌ Correctness (wrong directories indexed)
- ❌ Idempotency (state pollution)
- ❌ User trust (silent failures)

**L9 Verdict**: This is an unacceptable trade-off. Correctness > Performance.

## Comprehensive Analysis: How ADR-60 Broke Previous ADR Goals

### ADR Goal Achievement Analysis

#### ADR-44: Container Orchestration Architecture
**Goal**: Single source of truth, service discovery, dependency injection
**Status**: ❌ BROKEN
- ContainerDiscoveryService is initialized but NEVER USED
- No dependency injection - discovery service orphaned
- State still fragmented (each ensure_indexer creates own state)

#### ADR-48: Idempotent Container Management
**Goal**: Always remove existing containers before creating new ones
**Status**: ❌ COMPLETELY REMOVED
- ADR-60 does opposite - reuses existing containers
- No removal logic except 7-day garbage collection
- Lost idempotency guarantee

#### ADR-37: Configuration Priority Standard
**Goal**: Environment variables > config files > auto-detection
**Status**: ⚠️ UNTESTED
- Code might respect PROJECT_NAME env var
- But tests never verify this priority
- Container reuse doesn't check if env vars changed

#### ADR-30: Multi-Container Orchestration
**Goal**: Manage multiple project containers efficiently
**Status**: ⚠️ PARTIALLY BROKEN
- Can create multiple containers (different projects)
- But reuse logic breaks project isolation
- Wrong mounts = wrong project data

#### ADR-52: Automatic Indexer Initialization
**Goal**: Zero-step indexing, automatic project detection
**Status**: ❌ BROKEN BY MOUNT ISSUES
- Auto-start works but indexes wrong directory
- User thinks it's working (container running)
- But no files actually indexed

### The Methodology Change That Broke Everything

ADR-60 changed from **"Remove and Recreate"** to **"Discover and Reuse"** but:

1. **Didn't verify preconditions for reuse** (mount paths, env vars)
2. **Didn't use existing discovery infrastructure** (ContainerDiscoveryService)
3. **Didn't maintain idempotency guarantees** (ADR-48's core goal)
4. **Didn't test the actual outcome** (can files be indexed?)

### Critical Code Issues Found

```python
# ISSUE 1: ContainerDiscoveryService initialized but unused
self.discovery_service = ContainerDiscoveryService(self.docker_client)
# But ensure_indexer never calls discovery_service methods!

# ISSUE 2: Reuse without verification
existing_container = await self._discover_existing_container(project_name)
if existing_container:
    return existing_container['id']  # NO MOUNT CHECK!

# ISSUE 3: No removal of stale containers
# Only removes 7+ day old containers in GC
# Should remove containers with wrong configuration immediately
```

## L9-Standard Solution

### Immediate Fix: Mount Verification Before Reuse

```python
async def _ensure_indexer_internal(self, project_name: str, project_path: str) -> str:
    """
    L9 Fix: Verify mounts before reuse, remove if mismatched
    """
    existing_container = await self._discover_existing_container(project_name)

    if existing_container:
        container = self.docker_client.containers.get(existing_container['id'])

        # CRITICAL: Verify mount matches requested path
        mounts = container.attrs.get('Mounts', [])
        mount_valid = any(
            m['Source'] == os.path.abspath(project_path) and
            m['Destination'] == '/workspace'
            for m in mounts
        )

        if mount_valid:
            logger.info(f"[L9] Reusing container with verified mount: {project_path}")
            # Update tracking with verified path
            self.active_indexers[project_name] = {
                'container_id': existing_container['id'],
                'last_activity': datetime.now(),
                'project_path': project_path,
                'port': existing_container.get('port')
            }
            return existing_container['id']
        else:
            # Mount mismatch - remove and recreate
            logger.warning(f"[L9] Mount mismatch - removing container")
            logger.warning(f"  Expected: {project_path}")
            logger.warning(f"  Actual mounts: {[m['Source'] for m in mounts]}")
            container.remove(force=True)
            await self._invalidate_cache(project_name)

    # Create new container with correct mount
    return await self._create_container(project_name, project_path)
```

### Performance Optimization: Cache Mount Verification

```python
async def _verify_container_mount(self, container_id: str, expected_path: str) -> bool:
    """
    L9: Cache mount verification results to avoid repeated Docker API calls
    """
    cache_key = f"mount:{container_id}:{expected_path}"

    # Check Redis cache first (5-second TTL for mount verification)
    if self.redis_client:
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return cached == "valid"
        except Exception:
            pass  # Continue to actual verification

    # Perform actual verification
    container = self.docker_client.containers.get(container_id)
    mounts = container.attrs.get('Mounts', [])
    is_valid = any(
        m['Source'] == os.path.abspath(expected_path) and
        m['Destination'] == '/workspace'
        for m in mounts
    )

    # Cache result
    if self.redis_client:
        try:
            await self.redis_client.setex(
                cache_key, 5, "valid" if is_valid else "invalid"
            )
        except Exception:
            pass  # Cache is optional

    return is_valid
```

### Edge Case: Container Removal Failure Handling

```python
async def _safe_container_remove(self, container_id: str, max_retries: int = 3):
    """
    L9: Robust container removal with retry logic
    """
    for attempt in range(max_retries):
        try:
            container = self.docker_client.containers.get(container_id)
            container.remove(force=True)
            logger.info(f"[L9] Successfully removed container on attempt {attempt + 1}")
            return True
        except docker.errors.APIError as e:
            if "removal of container" in str(e) and "is already in progress" in str(e):
                # Container is already being removed, wait and verify
                await asyncio.sleep(1)
                try:
                    self.docker_client.containers.get(container_id)
                except docker.errors.NotFound:
                    return True  # Successfully removed
            elif attempt < max_retries - 1:
                logger.warning(f"[L9] Removal attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"[L9] Failed to remove container after {max_retries} attempts")
                # Last resort: Mark container for manual cleanup
                if container.status == 'running':
                    container.stop(timeout=5)
                raise
    return False
```

### Migration Path for Deployed Systems

```python
# migration_script.py - Run once during deployment
async def cleanup_stale_containers():
    """
    L9: One-time cleanup of all containers with wrong mounts
    """
    docker_client = docker.from_env()
    containers = docker_client.containers.list(
        all=True,
        filters={'label': 'com.l9.managed=true'}
    )

    removed_count = 0
    for container in containers:
        project = container.labels.get('com.l9.project')
        if not project:
            continue

        # Check if mount exists and is accessible
        mounts = container.attrs.get('Mounts', [])
        for mount in mounts:
            source = mount.get('Source')
            if source and not os.path.exists(source):
                logger.info(f"[MIGRATION] Removing container {container.name} - mount {source} doesn't exist")
                container.remove(force=True)
                removed_count += 1
                break

    logger.info(f"[MIGRATION] Removed {removed_count} stale containers")
    return removed_count
```

### Critical Test Addition: Path Change Validation

```python
async def test_mount_validation_on_path_change():
    """
    L9 Test: Verify containers are recreated when paths change
    This test would have caught the ADR-60 regression
    """
    orchestrator = IndexerOrchestrator()

    # Scenario 1: Create container for project at path A
    container1 = await orchestrator.ensure_indexer('neural-novelist', '/tmp/test-v1')

    # Scenario 2: Same project name, different path (simulates real usage)
    container2 = await orchestrator.ensure_indexer('neural-novelist', '/Users/dev/real-project')

    # CRITICAL ASSERTIONS:
    assert container1 != container2, "Container should be recreated for different path"

    # Verify new container has correct mount
    container = docker_client.containers.get(container2)
    mounts = container.attrs['Mounts']
    assert any(
        m['Source'] == '/Users/dev/real-project' and m['Destination'] == '/workspace'
        for m in mounts
    ), "New container must be mounted to requested path"

    # Verify old container was removed
    with pytest.raises(docker.errors.NotFound):
        docker_client.containers.get(container1)
```

## Lessons Learned

### 1. Test Realism > Test Coverage
- 100% code coverage means nothing if tests don't reflect usage
- One realistic test > ten synthetic tests

### 2. Happy Path Testing is Dangerous
- Tests that always succeed are worse than no tests
- They create false confidence that blocks real fixes

### 3. Container Orchestration Requires State Validation
- Never trust container state without verification
- Mount points, environment variables, and labels can drift

### 4. Validation Must Test Outcomes, Not Mechanisms
- Don't test "did the function run?"
- Test "can the user achieve their goal?"

## Action Items

### Immediate (Today)
- [ ] Implement mount verification fix (with fallback to remove-recreate)
- [ ] Add the 3 priority tests that would have caught this bug
- [ ] Run migration script to clean up existing stale containers
- [ ] Deploy hotfix to production with monitoring

### Short Term (This Week)
- [ ] Add performance caching for mount verification (Redis, 5s TTL)
- [ ] Implement robust container removal with retry logic
- [ ] Wire in ContainerDiscoveryService properly (architectural debt)
- [ ] Update CI/CD to require the 3 priority tests pass

### Long Term (This Month)
- [ ] Expand to full path mutation test suite (after basics work)
- [ ] Add user journey tests (real usage patterns)
- [ ] Consider chaos testing (only after fundamentals are solid)
- [ ] Add production telemetry for mount verification success rates

## Remaining Architectural Debt

### 1. ContainerDiscoveryService Integration (ADR-44)

The service exists but is completely unused. Full integration requires:

```python
# indexer_orchestrator.py - Complete the integration
async def _ensure_indexer_internal(self, project_name: str, project_path: str) -> str:
    # ... mount verification logic ...

    # CRITICAL: Actually USE the discovery service
    if self.discovery_service:
        # Discovery service should handle ALL container lifecycle
        container_info = await self.discovery_service.ensure_container(
            project_name=project_name,
            project_path=project_path,
            image='l9-neural-indexer:production',
            mounts={project_path: '/workspace'},
            labels={'com.l9.project': project_name},
            environment=self._get_environment_config()
        )

        # Update tracking
        self.active_indexers[project_name] = {
            'container_id': container_info['id'],
            'last_activity': datetime.now(),
            'project_path': project_path,
            'port': container_info['port']
        }

        return container_info['id']

    # Only fallback to direct creation if discovery service unavailable
    logger.warning("[ADR-44] Discovery service not available, using direct creation")
    return await self._create_container(project_name, project_path)
```

### 2. Configuration Priority Testing (ADR-37)

Must verify the complete priority chain works correctly:

```python
async def test_configuration_priority_chain():
    """
    ADR-37: Verify env vars > config files > auto-detection
    """
    orchestrator = IndexerOrchestrator()

    # Setup test environment
    test_project = "config-test"
    test_path = "/tmp/config-test"

    # Test 1: Environment variable takes precedence
    os.environ['PROJECT_NAME'] = 'env-override'
    os.environ['NEO4J_PASSWORD'] = 'env-password'

    # Create config file with different values
    with open('pyproject.toml', 'w') as f:
        f.write('[tool.l9]\nproject_name = "config-value"\nneo4j_password = "config-password"')

    container = await orchestrator.ensure_indexer(test_project, test_path)
    container_obj = docker_client.containers.get(container)
    env_vars = dict(e.split('=', 1) for e in container_obj.attrs['Config']['Env'])

    assert env_vars['PROJECT_NAME'] == 'env-override', "Env var should override config"
    assert env_vars['NEO4J_PASSWORD'] == 'env-password'

    # Test 2: Config file takes precedence over auto-detection
    del os.environ['PROJECT_NAME']  # Remove env var

    container2 = await orchestrator.ensure_indexer(test_project, test_path)
    container_obj2 = docker_client.containers.get(container2)
    env_vars2 = dict(e.split('=', 1) for e in container_obj2.attrs['Config']['Env'])

    assert env_vars2['PROJECT_NAME'] == 'config-value', "Config should override auto-detection"

    # Test 3: Auto-detection as last resort
    os.remove('pyproject.toml')  # Remove config file

    container3 = await orchestrator.ensure_indexer(test_project, test_path)
    container_obj3 = docker_client.containers.get(container3)
    env_vars3 = dict(e.split('=', 1) for e in container_obj3.attrs['Config']['Env'])

    assert env_vars3['PROJECT_NAME'] == test_project, "Should fall back to auto-detection"

    # Test 4: Container recreation on config change
    assert container != container2, "Config change should force recreation"
    assert container2 != container3, "Config removal should force recreation"
```

## Decision Outcome

**Accepted** - This regression exposed fundamental flaws in our testing philosophy. We must:

1. **Fix immediately** by restoring ADR-48's remove-before-create behavior
2. **Enhance validation** with path mutation and mount verification tests
3. **Shift philosophy** from mechanism testing to outcome testing
4. **Add chaos testing** to catch edge cases we can't anticipate

The cost of false confidence from inadequate tests far exceeds the cost of comprehensive validation.

## References

- ADR-0048: Idempotent Container Management (working implementation)
- ADR-0060: Graceful Ephemeral Containers (introduced regression)
- Production Issue: neural-novelist indexer only finding README.md
- Test Gap Analysis: All tests use /tmp/test path

## Testing Strategy (Updated September 21, 2025)

### Test Philosophy

After reviewing suggestions from various sources and considering our production incident, we establish the following testing principles:

1. **Critical Path Must Use Real Docker**: Mount validation CANNOT be mocked - Docker's actual mount behavior is what we're testing
2. **Performance vs Correctness**: We accept 10-12 second test runtime to prevent hours of debugging production issues  
3. **Test What Failed**: Every test must directly prevent the neural-novelist regression

### Test Boundaries

#### Must Use Real Docker Containers
- **Mount path validation**: The core regression - container reused with wrong mount
- **Environment variable propagation**: Actual Docker env var behavior
- **Container lifecycle**: Creation, reuse, cleanup
- **Label-based discovery**: Our ADR-060 discovery mechanism

#### Can Be Unit Tested (No Docker)
- **Label generation logic**: Test patterns for `com.l9.test=true`
- **Timestamp/random suffix generation**: Collision avoidance
- **Configuration parsing**: Environment variable extraction
- **Decision logic**: IF we refactor to pure functions (future work)

### Minimum Test Set to Prevent Regression

1. **test_critical_mount_validation.py** (3-4 seconds)
   - Creates 2 containers with different paths
   - Verifies each gets correct mount
   - Ensures no reuse with wrong path
   - **This test alone would have caught the regression**

2. **test_adr_63_mount_validation.py** (7-8 seconds)
   - Priority 1: Mount change forces recreation
   - Priority 2: Stale containers replaced
   - Priority 3: Env var changes trigger recreation
   - **Comprehensive edge case coverage**

3. **test_container_cleanup.py** (2-3 seconds) - PROPOSED
   - Verify test containers get `com.l9.test=true` label
   - Ensure cleanup removes all test containers
   - Prevent test pollution

### Why We Don't Mock Docker (September 2025 Decision)

After considering mocking strategies suggested by 2024-era systems, we decided AGAINST mocking for mount validation because:

1. **Docker's mount behavior is complex**: Symlinks, permissions, bind propagation modes
2. **The regression was in Docker interaction**: Not in our logic, but in how we used Docker
3. **Mock accuracy risk**: A mock might pass while real Docker fails
4. **10 seconds is acceptable**: For preventing production incidents

### Future Optimization Path (Not Current Priority)

IF test time becomes problematic (>30 seconds), consider:
1. Refactor validation logic to pure functions (as suggested)
2. Add unit tests for pure logic
3. Keep integration tests for critical paths
4. Use `alpine:latest` instead of full indexer image for tests

### Test Quality Metrics

- **Regression Prevention**: Would this test have caught neural-novelist issue? ✅
- **Cleanup Reliability**: Do test containers get removed? ✅
- **Time Budget**: Under 15 seconds total? ✅ (currently ~12s)
- **Flakiness**: Zero tolerance for flaky tests

### Edge Cases to Test (Validated September 2025)

From external review, these edge cases should be covered:

1. **Mount source doesn't exist** - Docker creates it, we should handle
2. **Permission issues** - Container can't read mount
3. **Empty vs unset env vars** - `VAR=""` vs no VAR
4. **Container states** - stopped, restarting, exited
5. **Image updates** - New image version should invalidate old containers
6. **Concurrent requests** - Redis locking prevents duplicates

Currently covered: 1, 3, 6
Need to add: 2, 4, 5
