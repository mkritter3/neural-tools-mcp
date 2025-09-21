# ADR-0064: Indexer Initialization Resilience & ADR-63 Test Hardening

**Status**: Proposed
**Date**: September 21, 2025
**Author**: Codex (on behalf of L9 Engineering)
**Related ADRs**: 0037, 0052, 0058, 0060, 0063

## Executive Summary

ADR-0058/ADR-0060 introduced Redis-backed distributed locking inside `IndexerOrchestrator.initialize()` to align with September 2025 container-orchestration guidance. In practice, the initialization now fails whenever Redis authentication, ACL, or transport settings drift from defaults. Because `ServiceContainer.ensure_indexer_running()` treats initialization failures as fatal, **no indexer containers start** for any project (including `neural-novelist`).

ADR-0063 attempted to add mount-validation tests that would have prevented the regression, but the suite depends on real Docker images and host access. CI and most developer laptops lack the `l9-neural-indexer:production` image and local daemon permissions, so the tests fail during setup and never exercise the critical assertions.

This ADR restores resilient initialization, codifies fallback behaviour, and replaces brittle ADR-63 tests with deterministic, mock-driven verification that runs in all environments.

## Problem Statement

1. **Redis handshake is mandatory**: `IndexerOrchestrator.initialize()` only catches `redis.ConnectionError`. When Redis replies with `AuthenticationError`, `ResponseError`, or TLS failures, the exception escapes, the orchestrator never sets up Docker discovery, and the indexer sidecar remains offline.
2. **Service container aborts immediately**: `ServiceContainer.ensure_indexer_running()` awaits `initialize()` inside the default control flow. One bubbled exception prevents reuse of the orchestrator, so every tool call re-triggers the same failure.
3. **ADR-63 tests are non-executable**: The suite launches live Docker containers. Missing images, restricted sockets, or sandboxed CI result in immediate setup errors, so regressions slip through.

## Goals

- Allow indexer startup to succeed even when Redis is unreachable or misconfigured, falling back to in-process locks while surfacing actionable telemetry.
- Guarantee Docker discovery and container lifecycle stay operational for all projects, including `neural-novelist`.
- Provide deterministic test coverage for mount-path and env-drift scenarios without requiring Docker.
- Keep optional integration coverage for full-stack validation when the environment supports it.

## Non-Goals

- Redesigning Redis locking semantics (we still prefer Redis when available).
- Changing external configuration defaults defined in ADR-0037 (e.g., `cache-secret-key`).
- Reverting ADR-0060’s container naming or label strategy.

## Detailed Plan

### 1. Harden `IndexerOrchestrator.initialize()`

1. Wrap Redis setup in a helper (`_init_redis()` or inline) that catches **all** `redis.RedisError` subclasses plus generic `Exception`.
2. On failure:
   - Log a structured warning: include host, port, username, and failure class (`AuthenticationError`, `TimeoutError`, etc.).
   - Increment a counter/metric if available (`logger` for now; follow-up for Prometheus hook).
   - Set `self.redis_client = None` so later code paths take the local `asyncio.Lock` fallback.
3. Proceed to initialize Docker discovery and schedule cleanup tasks even when Redis is down.
4. Keep successful Redis handshake behaviour unchanged (log success, start cache-aware tasks).

### 2. Guard `ensure_indexer_running()` Against Repeated Failures

1. Wrap the call to `initialize()` in `ServiceContainer.ensure_indexer_running()` with its own try/except.
2. If initialization fails:
   - Log the exception with context (project name, path).
   - Attempt a single retry **without** Redis by forcing `self.indexer_orchestrator.redis_client = None` and calling a new `_initialize_without_redis()` helper, or simply re-invoking `initialize()` after toggling a flag.
   - If the retry still fails, surface a structured error that clearly states the indexer was not started but avoid re-raising raw Redis exceptions to the MCP tool layer.
3. Cache the fact that initialization succeeded (or the fallback succeeded) to avoid repeat retries on every tool call.

### 3. Refactor ADR-63 Tests

1. Split the current file into:
   - `tests/test_indexer_mount_validation.py`: mock-based unit tests.
   - `tests/integration/test_indexer_mount_validation.py`: optional integration tests (guarded by `pytest.mark.docker` and skipped when Docker isn’t available or the image is missing).
2. For the unit suite:
   - Patch `IndexerOrchestrator.docker_client` with a `MagicMock` that mimics `containers.list`, `containers.get`, and `containers.run` responses.
   - Simulate mount/env mismatches by altering the mocked container attrs across calls, then assert that `_ensure_indexer_internal` removes and recreates containers when paths or env vars change.
   - Validate cache invalidation and environment comparisons.
3. Provide fixtures/helpers that create deterministic container metadata dictionaries (source path, env list, labels).
4. Ensure the tests run in <1s on CI and developer machines without Docker.

### 4. Add Integration Guard (Optional)

1. Retain a thin integration test that only runs when:
   - `docker.from_env()` succeeds.
   - The `l9-neural-indexer:production` image is present (skip otherwise).
   - Redis credentials are available.
2. Use `pytest.skip()` with clear messaging when prerequisites are missing, so the default pipeline never blocks.

### 5. Observability & Alerts

1. Add a `logger.warning` in `ServiceContainer.ensure_indexer_running()` when falling back to local locks, including the Redis failure type.
2. Expose a lightweight status hook (e.g., `get_status()` or health endpoint metric) tracking `redis_lock_available` so operators know when the system is on the fallback path.
3. Consider (follow-up ticket) surfacing this flag through MCP’s `neural_system_status` tool.

## Acceptance Criteria

- Indexer sidecar starts for `neural-novelist` with Redis credentials intentionally set to an invalid value.
- Running `pytest tests/test_indexer_mount_validation.py` succeeds on a clean checkout without Docker installed.
- Integration suite (`pytest tests/integration/test_indexer_mount_validation.py`) either passes or skips with a descriptive message depending on Docker availability.
- Log output shows a single warning when Redis auth is wrong and confirms fallback to local locking; no stack trace leaks to the MCP client.

## Rollout Strategy

1. Implement code changes behind feature branches; run full unit suite locally.
2. Update CI to include the new unit tests and optional integration marker.
3. Deploy to staging MCP instance; simulate Redis auth failure and verify the indexer still starts while telemetry records the warning.
4. Monitor production for 48 hours; track counts of fallback warnings. If the rate remains acceptable, close out the incident.

## Alternatives Considered

- **Strict failure on Redis errors**: rejected because it reintroduces the current outage mode.
- **Environment-based opt-in for Redis**: rejected; risk of drifting config across environments. Graceful degradation is preferred.
- **Removing Redis locking entirely**: rejected; we still benefit from cross-instance coordination when Redis is healthy.

## Follow-Up Work

- Enhance `DiscoveryService` coverage using the same mock strategy (future ADR).
- Add Prometheus counters for fallback activations (tie into ADR-0054 observability workstream).
- Re-run ADR-60 end-to-end tests once the suite is updated to mock Redis failures explicitly.

