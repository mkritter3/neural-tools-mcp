# ADR-0045: Neural Tools CI/CD and Deployment Process

**Status**: Approved
**Date**: September 13, 2025
**Author**: L9 Engineering Team

## Context

Following the user's concern about deployment practices ("can you stop deploying without testing or ci/cd?"), we need to establish a robust CI/CD pipeline and deployment process for neural-tools. This ensures all changes are properly tested before reaching production.

## Problem Statement

Current deployment issues:
1. Manual deployment without automated testing
2. No validation before global MCP deployment
3. No rollback mechanism for failed deployments
4. No breaking change detection
5. Lack of deployment audit trail

## Decision

Implement a comprehensive CI/CD pipeline with:
1. **GitHub Actions workflow** for automated testing
2. **Pre-deployment validation** scripts
3. **Breaking change detection**
4. **Staged deployment process**
5. **Automated rollback capability**

## Implementation

### 1. CI/CD Pipeline Structure

```yaml
Pipeline Stages:
├── Lint & Type Check
├── Unit Tests
├── Integration Tests
├── ADR Validation
├── Pre-Deployment Validation
└── Deploy to Global MCP
```

### 2. Testing Requirements

**Before ANY deployment:**
- ✅ Python syntax validation
- ✅ Import integrity checks
- ✅ ADR compliance verification
- ✅ Unit test coverage >80%
- ✅ Integration tests with real services
- ✅ Breaking change detection
- ✅ Docker image validation (ADR-0038)

### 3. Deployment Process

```bash
# Development workflow
1. Make changes in project directory
2. Run local tests: ./neural-tools/scripts/run-tests.sh
3. Commit to feature branch
4. CI runs on push
5. Create PR for review
6. CI validates PR
7. Merge to main
8. Automatic deployment validation
9. Manual approval for production
10. Deploy to global MCP
```

### 4. Validation Scripts

**validate-deployment.py**: Pre-deployment checks
- Python syntax validation
- Import resolution
- ADR implementation verification
- Configuration validation
- Test coverage check
- Docker image compliance
- Dependency verification

**check-breaking-changes.py**: API compatibility
- Removed functions detection
- Signature changes
- Import structure changes
- Configuration changes
- MCP tool changes

**run-tests.sh**: Comprehensive test runner
- Environment checks
- Service availability
- Unit tests
- Integration tests
- ADR validation
- Deployment validation

### 5. GitHub Actions Workflow

```yaml
# .github/workflows/neural-tools-ci.yml
on:
  push:
    paths: ['neural-tools/**', 'docs/adr/**']
  pull_request:
    paths: ['neural-tools/**', 'docs/adr/**']

jobs:
  lint:           # Code quality checks
  unit-tests:     # Unit test suite
  integration:    # Integration with services
  adr-validation: # ADR compliance
  deploy-check:   # Pre-deployment validation
  deploy:         # Production deployment (main only)
```

### 6. Service Testing

Integration tests run against real services:
- Neo4j on port 47687
- Qdrant on port 46333
- Redis on port 46379

Tests use GitHub Actions services for isolation.

### 7. Deployment Manifest

Each deployment generates a manifest:
```json
{
  "version": "git-sha",
  "timestamp": "2025-09-13T10:30:00Z",
  "tests_passed": ["unit", "integration", "adr"],
  "breaking_changes": [],
  "deployment_type": "production",
  "rollback_point": "previous-sha"
}
```

### 8. Rollback Process

If deployment fails:
```bash
# Automatic rollback triggered by post-deployment validation
1. Detect failure in post-deployment check
2. Restore from backup (created pre-deployment)
3. Notify team of rollback
4. Generate incident report
```

## Testing Strategy

### Local Testing (Development)
```bash
# Run before committing
cd neural-tools
./scripts/run-tests.sh

# Quick validation
python scripts/validate-deployment.py

# Check breaking changes
python scripts/check-breaking-changes.py
```

### CI Testing (Automated)
- Every push triggers linting and unit tests
- PRs trigger full test suite
- Main branch triggers deployment validation
- Production deployment requires manual approval

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Component isolation
   - Mock external dependencies
   - Fast execution (<30s)

2. **Integration Tests** (`tests/integration/`)
   - Real service connections
   - End-to-end workflows
   - Container orchestration

3. **ADR Validation** (`tests/validation/`)
   - ADR-0043: Project lifecycle
   - ADR-0044: Container orchestration
   - ADR-0037: Configuration priority

4. **Deployment Tests**
   - Pre-deployment validation
   - Breaking change detection
   - Post-deployment health checks

## Deployment Environments

### Development
- Local `.mcp.json` configuration
- Direct file editing
- Immediate testing

### Staging (CI)
- GitHub Actions runners
- Dockerized services
- Isolated testing

### Production
- Global MCP (`~/.claude/mcp-servers/`)
- Manual approval required
- Automated rollback on failure

## Success Metrics

- **Zero** untested deployments
- **<5%** deployment failure rate
- **<5 min** rollback time
- **100%** ADR compliance
- **>80%** test coverage

## Consequences

### Positive
- ✅ No more untested deployments
- ✅ Automatic breaking change detection
- ✅ Fast rollback capability
- ✅ Clear deployment audit trail
- ✅ Confidence in production deployments

### Negative
- ❌ Slightly slower deployment process
- ❌ Requires GitHub Actions setup
- ❌ More complex deployment workflow

### Mitigations
- Parallel test execution for speed
- Local test runner for quick validation
- Clear documentation and scripts

## Compliance

This ADR ensures compliance with:
- **L9 Engineering Standards**: 95% gate for risky changes
- **ADR-0038**: Docker image lifecycle management
- **ADR-0037**: Configuration priority standard
- **Truth-First Contract**: Evidence-based deployment

## Example Usage

### Running Tests Locally
```bash
# Full test suite
./neural-tools/scripts/run-tests.sh

# Just validation
python neural-tools/scripts/validate-deployment.py

# Check for breaking changes
python neural-tools/scripts/check-breaking-changes.py
```

### Deployment Command
```bash
# After tests pass
./scripts/deploy-to-global-mcp.sh

# Script now includes:
# 1. Pre-deployment validation
# 2. Backup creation
# 3. Deployment
# 4. Post-deployment validation
# 5. Automatic rollback on failure
```

## Status Tracking

- [x] CI/CD pipeline configuration created
- [x] Pre-deployment validation script
- [x] Breaking change detection
- [x] Automated test runner
- [x] GitHub Actions workflow
- [ ] Integration with deploy-to-global-mcp.sh
- [ ] Post-deployment validation
- [ ] Automatic rollback mechanism

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [MCP Protocol 2025-06-18](https://modelcontextprotocol.io/)
- ADR-0043: Project Context Lifecycle
- ADR-0044: Container Orchestration