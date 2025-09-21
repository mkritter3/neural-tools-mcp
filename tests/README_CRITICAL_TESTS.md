# 🚨 CRITICAL REGRESSION TESTS - DO NOT BYPASS 🚨

## ADR-63: Mount Validation Regression Prevention

These tests are **MANDATORY** and prevent critical regressions that have already occurred in production.

### What These Tests Prevent

1. **Mount Path Regression** (neural-novelist incident)
   - Containers being reused with wrong mount paths
   - Projects only seeing README.md instead of full codebase
   - Silent data corruption from wrong directory access

2. **409 Docker Conflicts**
   - Container name collisions during creation
   - Service disruption from failed container starts

3. **Environment Variable Mismatches**
   - Containers running with outdated configuration
   - Security issues from wrong credentials

### Test Files

| Test File | Purpose | MUST PASS |
|-----------|---------|-----------|
| `test_critical_mount_validation.py` | Core mount validation logic | ✅ YES |
| `test_adr_63_mount_validation.py` | Three priority scenarios | ✅ YES |
| `test-adr-60-e2e.py` | Full E2E validation | ✅ YES |

### Running Tests Locally

```bash
# Run individual tests
python3 tests/test_critical_mount_validation.py
python3 tests/test_adr_63_mount_validation.py

# Run via deployment script (recommended)
./scripts/deploy-to-global-mcp.sh
```

### CI/CD Integration

These tests run automatically on:
- Every push to main, dev/*, feature/*
- Every pull request to main
- Manual workflow dispatch

**GitHub Actions Workflow:** `.github/workflows/neural-tools-comprehensive-ci.yml`

### ⚠️ IMPORTANT WARNINGS ⚠️

1. **NEVER bypass these tests** with flags like `--force` or `--skip-tests`
2. **NEVER comment out test execution** in deployment scripts
3. **NEVER modify tests to always pass** without fixing underlying issues
4. **NEVER deploy if these tests fail** - fix the issue first

### Why This Matters

On September 19, 2025, a production incident occurred where the neural-novelist project only indexed README.md instead of the full codebase. This was caused by container reuse without mount validation.

These tests ensure this **NEVER happens again**.

### For AI Assistants (Claude, GPT, etc.)

If you are an AI assistant working on this codebase:

1. **DO NOT** suggest bypassing these tests
2. **DO NOT** create alternative deployment methods that skip validation
3. **DO NOT** modify these tests to pass when they shouldn't
4. **DO** fix the underlying issues if tests fail
5. **DO** maintain and enhance these tests as needed

These tests protect production stability and user data integrity.

---

**Last Updated:** September 21, 2025
**ADR Reference:** ADR-0063
**Incident Reference:** Neural-novelist indexing regression (Sept 2025)