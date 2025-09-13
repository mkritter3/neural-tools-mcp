# ADR-0037 Team Guide: Container Configuration Priority Standard

**Complete guide for L9 engineering teams on implementing and maintaining the ADR-0037 container configuration standard.**

---

## 📚 Table of Contents

- [Quick Start](#-quick-start)
- [Understanding ADR-0037](#-understanding-adr-0037)
- [Developer Workflow](#-developer-workflow)
- [Common Scenarios](#-common-scenarios)
- [Troubleshooting](#-troubleshooting)
- [Best Practices](#-best-practices)
- [Reference](#-reference)

---

## 🚀 Quick Start

### For New Developers

1. **Read the ADR**: Start with `docs/adr/0037-indexer-container-configuration-priority.md`
2. **Run validation**: `python3 scripts/validate-adr-0037.py --all`
3. **Check code patterns**: Use the code review checklist before submitting PRs

### For Existing Projects

1. **Validate current setup**: Run `./scripts/ci-validate-adr-0037.sh`
2. **Fix any violations**: Follow the error messages and examples
3. **Update containers**: Ensure environment variables are properly set

### Essential Commands

```bash
# Validate all containers
python3 scripts/validate-adr-0037.py --all

# Validate specific container
python3 scripts/validate-adr-0037.py --container [NAME]

# Validate code changes
python3 scripts/validate-adr-0037.py --validate-code [DIRECTORY]

# CI/CD validation
./scripts/ci-validate-adr-0037.sh
```

---

## 🎯 Understanding ADR-0037

### The Problem It Solves

**Before ADR-0037:**
```bash
# Container ignores explicit configuration
docker run -e PROJECT_NAME=my-project my-container
# Still auto-detects wrong directory: "docker" instead of "my-project"
```

**After ADR-0037:**
```bash
# Container respects explicit configuration
docker run -e PROJECT_NAME=my-project my-container  
# Uses explicit value: "my-project" ✅
```

### Core Principle: Configuration Priority Hierarchy

1. **Environment Variables** (Highest Priority)
   - Explicit, predictable, follows Twelve-Factor App
   - Required for production deployments

2. **Configuration Files** (Medium Priority)
   - `pyproject.toml`, `package.json`, `.graphrag/config.yaml`
   - Good for development defaults

3. **Auto-Detection** (Fallback Only)
   - Project markers, directory names
   - Should not be relied upon in production

4. **Hard-coded Defaults** (Last Resort)
   - `DEFAULT_PROJECT_NAME = "default"`
   - Safety net only

### Why This Matters

- **🏥 Production Reliability**: Prevents "works on my machine" issues
- **🔄 Multi-Project Support**: Enables proper project isolation
- **🚀 Scalability**: Standardizes configuration across all services
- **🛡️ Security**: Explicit configuration prevents accidental data mixing

---

## 👨‍💻 Developer Workflow

### Creating New Containerized Services

#### 1. Design Configuration Interface

```python
# REQUIRED: Support environment variables first
project_name = os.getenv("PROJECT_NAME")
project_path = os.getenv("PROJECT_PATH")

if project_name and project_path:
    logger.info(f"✅ [ADR-0037] Using explicit config: {project_name}")
    # Use explicit configuration
else:
    logger.warning("⚠️ [ADR-0037] Falling back to auto-detection")
    # Auto-detection fallback
```

#### 2. Container Environment Setup

```dockerfile
# For project-aware containers (indexers, processors)
ENV PROJECT_NAME=claude-l9-template
ENV PROJECT_PATH=/workspace

# For container-to-host communication  
ENV NEO4J_URI=bolt://host.docker.internal:47687
ENV QDRANT_HOST=host.docker.internal
ENV QDRANT_PORT=46333
```

#### 3. Validation During Development

```bash
# Before committing
python3 scripts/validate-adr-0037.py --validate-code [YOUR_CODE_DIR]

# Before deploying
python3 scripts/validate-adr-0037.py --container [YOUR_CONTAINER]
```

### Modifying Existing Services

#### 1. Assess Current Compliance

```bash
# Check if service is compliant
python3 scripts/validate-adr-0037.py --container [EXISTING_CONTAINER]
```

#### 2. Common Fixes Needed

**Fix 1: Add Environment Variable Priority**
```python
# BEFORE (non-compliant)
project_name = detect_from_directory()

# AFTER (compliant)
project_name = os.getenv("PROJECT_NAME")
if not project_name:
    project_name = detect_from_directory()
    logger.warning("⚠️ [ADR-0037] Using auto-detection fallback")
```

**Fix 2: Update Container Networking**
```bash
# BEFORE (non-compliant)
NEO4J_URI=bolt://localhost:47687

# AFTER (compliant)  
NEO4J_URI=bolt://host.docker.internal:47687
```

**Fix 3: Add Error Handling**
```python
# BEFORE (non-compliant)
project_path = Path(os.getenv("PROJECT_PATH", "/workspace"))

# AFTER (compliant)
project_path_env = os.getenv("PROJECT_PATH")
if project_path_env:
    project_path = Path(project_path_env)
    if not project_path.exists():
        logger.error(f"PROJECT_PATH does not exist: {project_path_env}")
        sys.exit(1)
else:
    project_path = auto_detect_path()
```

---

## 📋 Common Scenarios

### Scenario 1: Local Development

```bash
# Run with explicit project configuration
docker run \
  -e PROJECT_NAME=my-local-project \
  -e PROJECT_PATH=/workspace \
  -v $(pwd):/workspace:ro \
  my-indexer:latest
```

### Scenario 2: Multi-Project Environment

```bash
# Project A
docker run \
  -e PROJECT_NAME=project-a \
  -e PROJECT_PATH=/workspace \
  -v /path/to/project-a:/workspace:ro \
  my-indexer:latest

# Project B  
docker run \
  -e PROJECT_NAME=project-b \
  -e PROJECT_PATH=/workspace \
  -v /path/to/project-b:/workspace:ro \
  my-indexer:latest
```

### Scenario 3: Testing Auto-Detection

```bash
# Test fallback behavior (no env vars)
docker run \
  -v $(pwd):/workspace:ro \
  my-indexer:latest
  
# Should see: "⚠️ [ADR-0037] Falling back to auto-detection"
```

### Scenario 4: CI/CD Integration

```yaml
# GitHub Actions
- name: Validate ADR-0037 Compliance
  run: ./scripts/ci-validate-adr-0037.sh --code-only

# Docker build
- name: Test Container Configuration
  run: |
    docker build -t test-image .
    docker run --rm \
      -e PROJECT_NAME=ci-test \
      -e PROJECT_PATH=/workspace \
      test-image validate-config
```

---

## 🔧 Troubleshooting

### Common Error Messages

#### "Missing required environment variable 'PROJECT_NAME'"

**Problem**: Container launched without required env vars
```bash
# ❌ WRONG
docker run my-indexer:latest

# ✅ CORRECT  
docker run -e PROJECT_NAME=my-project -e PROJECT_PATH=/workspace my-indexer:latest
```

#### "Using localhost instead of host.docker.internal"

**Problem**: Container networking misconfigured
```python
# ❌ WRONG (in container code)
NEO4J_URI = "bolt://localhost:47687"

# ✅ CORRECT (in container code)
NEO4J_URI = "bolt://host.docker.internal:47687"
```

#### "Using filesystem order instead of env vars"

**Problem**: Code doesn't check environment variables first
```python
# ❌ WRONG
selected_dir = directories[0]  # Filesystem order!

# ✅ CORRECT
project_name = os.getenv("PROJECT_NAME")
if not project_name:
    selected_dir = smart_detection_with_markers()
```

### Debugging Steps

1. **Check Environment Variables**
   ```bash
   docker inspect [CONTAINER] | jq '.[0].Config.Env'
   ```

2. **Validate Configuration**
   ```bash
   python3 scripts/validate-adr-0037.py --container [CONTAINER]
   ```

3. **Check Container Logs**
   ```bash
   docker logs [CONTAINER] | grep "ADR-0037"
   ```

4. **Test Explicit Configuration**
   ```bash
   # Test with explicit env vars
   docker run -e PROJECT_NAME=test -e PROJECT_PATH=/workspace [IMAGE]
   ```

---

## ✅ Best Practices

### Code Quality

1. **Always Reference ADR-0037 in Comments**
   ```python
   # ADR-0037: Environment Variable Configuration Priority
   project_name = os.getenv("PROJECT_NAME")
   ```

2. **Provide Clear Logging**
   ```python
   if explicit_config:
       logger.info(f"✅ [ADR-0037] Using explicit config: {project_name}")
   else:
       logger.warning("⚠️ [ADR-0037] Falling back to auto-detection")
   ```

3. **Validate Explicit Configuration**
   ```python
   if project_path and not Path(project_path).exists():
       logger.error(f"[ADR-0037] Invalid PROJECT_PATH: {project_path}")
       sys.exit(1)
   ```

### Container Design

1. **Separate Infrastructure from Project-Aware Services**
   - **Infrastructure**: Neo4j, Qdrant, Redis (no PROJECT_NAME needed)
   - **Project-Aware**: Indexers, processors (require PROJECT_NAME)

2. **Use Consistent Environment Variable Names**
   - `PROJECT_NAME`: Project identifier
   - `PROJECT_PATH`: Project root directory
   - `NEO4J_URI`: Neo4j connection string
   - `QDRANT_HOST`: Qdrant hostname

3. **Document Environment Variables**
   ```dockerfile
   # ADR-0037 Required Variables
   ENV PROJECT_NAME="" \
       PROJECT_PATH="/workspace"
   
   # Service Connection Variables  
   ENV NEO4J_URI="bolt://host.docker.internal:47687" \
       QDRANT_HOST="host.docker.internal"
   ```

### Testing

1. **Always Test Both Modes**
   ```bash
   # Test explicit configuration
   docker run -e PROJECT_NAME=test -e PROJECT_PATH=/workspace [IMAGE]
   
   # Test auto-detection fallback
   docker run [IMAGE]
   ```

2. **Include Validation in CI/CD**
   ```yaml
   - name: ADR-0037 Validation
     run: ./scripts/ci-validate-adr-0037.sh
   ```

3. **Use Validation Before Code Reviews**
   ```bash
   # Before submitting PR
   python3 scripts/validate-adr-0037.py --validate-code .
   ```

---

## 📖 Reference

### Files and Tools

| File | Purpose |
|------|---------|
| `docs/adr/0037-*.md` | Official ADR document |
| `scripts/validate-adr-0037.py` | Validation utility |
| `scripts/ci-validate-adr-0037.sh` | CI/CD integration script |
| `docs/ADR-0037-CODE-REVIEW-CHECKLIST.md` | Code review guidelines |
| `.github/workflows/adr-0037-validation.yml` | GitHub Actions workflow |

### Environment Variables Reference

#### Required for Project-Aware Services
- `PROJECT_NAME`: Project identifier (e.g., `claude-l9-template`)
- `PROJECT_PATH`: Project root directory (e.g., `/workspace`)

#### Container-to-Host Communication
- `NEO4J_URI`: `bolt://host.docker.internal:47687`
- `NEO4J_PASSWORD`: `graphrag-password`
- `QDRANT_HOST`: `host.docker.internal`
- `QDRANT_PORT`: `46333`
- `EMBEDDING_SERVICE_HOST`: `host.docker.internal`
- `EMBEDDING_SERVICE_PORT`: `48000`
- `REDIS_CACHE_HOST`: `host.docker.internal`
- `REDIS_CACHE_PORT`: `46379`
- `REDIS_QUEUE_HOST`: `host.docker.internal`
- `REDIS_QUEUE_PORT`: `46380`

### Code Examples

#### Compliant Configuration Loading
```python
def load_project_configuration():
    """Load project configuration following ADR-0037 priority hierarchy"""
    
    # 1. PRIORITY: Environment Variables
    project_name = os.getenv("PROJECT_NAME")
    project_path = os.getenv("PROJECT_PATH")
    
    if project_name and project_path:
        if not os.path.isdir(project_path):
            logging.error(f"PROJECT_PATH does not exist: {project_path}")
            sys.exit(1)
        logging.info(f"✅ [ADR-0037] Using explicit config: {project_name}")
        return project_name, project_path
    
    # 2. FALLBACK: Auto-detection (only if env vars missing)
    logging.warning("⚠️ [ADR-0037] Environment variables missing. Falling back to auto-detection.")
    detected_name, detected_path = auto_detect_project_smart()
    
    project_name = project_name or detected_name
    project_path = project_path or detected_path
    
    logging.info(f"🔍 [ADR-0037] Auto-detected: {project_name}")
    return project_name, project_path
```

#### Compliant Docker Run Command
```bash
docker run -d \
  --name my-indexer \
  --network l9-graphrag-network \
  -e PROJECT_NAME=claude-l9-template \
  -e PROJECT_PATH=/workspace \
  -e NEO4J_URI=bolt://host.docker.internal:47687 \
  -e NEO4J_PASSWORD=graphrag-password \
  -e QDRANT_HOST=host.docker.internal \
  -e QDRANT_PORT=46333 \
  -v $(pwd):/workspace:ro \
  my-indexer:latest
```

---

## 🎓 Training Checklist

**Complete this checklist to ensure team members understand ADR-0037:**

### Understanding
- [ ] Read the complete ADR-0037 document
- [ ] Understand the configuration priority hierarchy
- [ ] Know why environment variables take precedence
- [ ] Understand container vs host networking differences

### Practical Skills
- [ ] Can run the validation utility successfully
- [ ] Can interpret validation errors and warnings
- [ ] Can write ADR-0037 compliant configuration code
- [ ] Can set up containers with proper environment variables

### Code Review
- [ ] Familiar with the code review checklist
- [ ] Can identify non-compliant patterns in code
- [ ] Understands when to use `host.docker.internal` vs `localhost`
- [ ] Can validate changes before submitting PRs

### Troubleshooting
- [ ] Can debug common configuration issues
- [ ] Knows how to check container environment variables
- [ ] Can test both explicit config and auto-detection modes
- [ ] Understands error messages and resolution steps

---

**Questions or need help?** Contact the L9 Engineering team or check the validation tool: `python3 scripts/validate-adr-0037.py --help`

**Remember**: ADR-0037 prevents critical production bugs. It's better to be strict about compliance than debug container startup issues later! 🛡️