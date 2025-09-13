# ADR-0037 Code Review Checklist
## Container Configuration Priority Standard

**Use this checklist for all code reviews involving containerized services or configuration changes.**

---

## 📋 Pre-Review Validation

**Before reviewing code, run automated validation:**

```bash
# Validate all containers
python3 scripts/validate-adr-0037.py --all

# Validate specific container (if testing new service)
python3 scripts/validate-adr-0037.py --container [CONTAINER_NAME]

# Validate code patterns in changed files
python3 scripts/validate-adr-0037.py --validate-code [DIRECTORY]
```

**✅ REQUIRED**: All validation must pass before code review approval.

---

## 🔍 Code Review Checklist

### Container Configuration Changes

- [ ] **Environment Variable Priority**: Does the code check environment variables FIRST?
  ```python
  ✅ CORRECT:
  project_name = os.getenv("PROJECT_NAME")
  if project_name:
      # Use explicit configuration
  else:
      # Fall back to auto-detection
  
  ❌ WRONG:
  # Auto-detection without checking env vars first
  project_name = detect_from_directory()
  ```

- [ ] **ADR-0037 Logging**: Are configuration decisions clearly logged?
  ```python
  ✅ CORRECT:
  logger.info(f"✅ [ADR-0037] Using explicit config: {project_name}")
  logger.warning("⚠️ [ADR-0037] Falling back to auto-detection")
  ```

- [ ] **Error Handling**: Does the code fail fast when explicit config is invalid?
  ```python
  ✅ CORRECT:
  if not project_path.exists():
      logger.error(f"PROJECT_PATH does not exist: {project_path}")
      sys.exit(1)
  ```

### Docker Configuration Changes

- [ ] **Host Networking**: Does container use `host.docker.internal` (not `localhost`)?
  ```bash
  ✅ CORRECT: NEO4J_URI=bolt://host.docker.internal:47687
  ❌ WRONG: NEO4J_URI=bolt://localhost:47687
  ❌ WRONG: NEO4J_URI=bolt://172.18.0.5:7687  # Docker internal IP
  ```

- [ ] **Required Environment Variables**: Are PROJECT_NAME/PROJECT_PATH set for project-aware services?
  ```dockerfile
  ✅ CORRECT (for indexer containers):
  ENV PROJECT_NAME=claude-l9-template
  ENV PROJECT_PATH=/workspace
  ```

- [ ] **Docker Compose**: Are environment variables properly defined?
  ```yaml
  ✅ CORRECT:
  environment:
    - PROJECT_NAME=claude-l9-template
    - PROJECT_PATH=/workspace
    - NEO4J_URI=bolt://host.docker.internal:47687
  ```

### Code Pattern Validation

- [ ] **No Hardcoded Assumptions**: Code doesn't assume specific directory structure?
  ```python
  ❌ WRONG:
  directories = os.listdir("/workspace")
  selected_dir = directories[0]  # Filesystem order dependency!
  
  ✅ CORRECT:
  project_name = os.getenv("PROJECT_NAME")
  if not project_name:
      # Smart auto-detection with project markers
  ```

- [ ] **Project Marker Detection**: Auto-detection uses project markers (not directory order)?
  ```python
  ✅ CORRECT:
  for marker in ['.git', 'pyproject.toml', 'package.json']:
      if (workspace_path / marker).exists():
          return workspace_path
  ```

### Documentation Requirements

- [ ] **ADR References**: Does code reference ADR-0037 in comments where applicable?
  ```python
  ✅ CORRECT:
  # ADR-0037: Environment Variable Configuration Priority
  project_name = os.getenv("PROJECT_NAME")
  ```

- [ ] **Configuration Examples**: Are examples updated to show compliant patterns?
- [ ] **Environment Variable Documentation**: Are new env vars documented in CLAUDE.md?

---

## 🚨 Automatic Rejection Criteria

**The following patterns require immediate code change requests:**

### ❌ Critical Violations (Block Merge)

1. **Using `localhost` in container configuration**
   ```bash
   NEO4J_URI=bolt://localhost:47687  # ❌ BLOCKS MERGE
   ```

2. **Ignoring environment variables**
   ```python
   # ❌ BLOCKS MERGE - No env var check
   project_name = os.path.basename(os.getcwd())
   ```

3. **Filesystem order dependency**
   ```python
   # ❌ BLOCKS MERGE - Directory order assumption
   selected_dir = directories[0]
   ```

4. **Docker internal IP hardcoding**
   ```python
   # ❌ BLOCKS MERGE - Hardcoded Docker IP
   host = "172.18.0.5"
   ```

### ⚠️ Warning Violations (Request Changes)

1. **Missing ADR-0037 logging**
2. **No error handling for invalid explicit config**
3. **Missing environment variable documentation**
4. **No validation test coverage**

---

## 🧪 Testing Requirements

**For container configuration changes, require:**

- [ ] **Validation Test**: `python3 scripts/validate-adr-0037.py` passes
- [ ] **Container Test**: New container starts with explicit env vars
- [ ] **Fallback Test**: Auto-detection works when env vars missing
- [ ] **Error Test**: Graceful failure when explicit config invalid

### Example Test Commands
```bash
# Test explicit configuration
docker run -e PROJECT_NAME=test -e PROJECT_PATH=/workspace [IMAGE]

# Test auto-detection fallback  
docker run [IMAGE]  # No env vars

# Test error handling
docker run -e PROJECT_NAME=test -e PROJECT_PATH=/nonexistent [IMAGE]
```

---

## 📚 Quick Reference Links

- **ADR Document**: `docs/adr/0037-indexer-container-configuration-priority.md`
- **Validation Tool**: `scripts/validate-adr-0037.py --help`
- **Configuration Standards**: `CLAUDE.md` → "ADR-0037: Container Configuration Priority Standard"
- **Example Implementation**: `docker/scripts/indexer-entrypoint.py` (lines 272-313)

---

## ✅ Review Approval Criteria

**Only approve when ALL of the following are met:**

1. ✅ `python3 scripts/validate-adr-0037.py` reports 0 errors
2. ✅ No critical violations present in code
3. ✅ Environment variables properly prioritized
4. ✅ Container networking uses `host.docker.internal`
5. ✅ ADR-0037 logging implemented where applicable
6. ✅ Documentation updated for new environment variables
7. ✅ Test coverage includes configuration scenarios

**Remember**: ADR-0037 prevents critical production bugs. Better to be strict during review than debug container startup issues later.

---

**Questions?** Run `python3 scripts/validate-adr-0037.py --help` or check the ADR document for implementation details.