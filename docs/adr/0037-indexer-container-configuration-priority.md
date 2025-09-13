# ADR-0037: Indexer Container Configuration Priority

**Status**: Proposed  
**Date**: September 12, 2025  
**Deciders**: L9 Engineering Team, Multi-Model Expert Consensus  
**Technical Story**: Critical indexer container startup bug violates Twelve-Factor App methodology

## Context

### Problem Discovery

During investigation of hybrid GraphRAG search failures, we discovered a critical bug in the indexer container startup logic:

1. **Environment variables ignored**: Container ignores explicit `PROJECT_NAME=claude-l9-template` and `PROJECT_PATH=/workspace`
2. **Fragile auto-detection**: Uses directory order to select `/workspace/docker` instead of `/workspace`  
3. **Wrong project scope**: Creates `project_docker_code` collection instead of `project-claude-l9-template`
4. **System impact**: Prevents hybrid search from accessing 7,366 existing Neo4j nodes

### Root Cause Analysis

The indexer container's `entrypoint.py` contains flawed startup logic:

```python
# BROKEN LOGIC (current implementation)
if multiple_directories_found:
    selected_dir = directories[0]  # Uses filesystem order!
    project_name = os.path.basename(selected_dir)  # Ignores env var!
```

**Log Evidence:**
```
WARNING: Multiple directories found: ['docker', 'memory', 'config', 'tests', ...]
WARNING: Using the first one found: docker
INFO: Auto-detected PROJECT_NAME='docker' from '/workspace/docker'
```

### Industry Standards Violation

This violates the **Twelve-Factor App methodology** (Factor III: Config):
- Configuration should come from environment variables
- Code should not make implicit assumptions about deployment context
- Explicit configuration takes priority over auto-detection

## Decision

**We will establish a standardized configuration priority hierarchy for all containerized services:**

### Configuration Priority Order

1. **Environment Variables** (highest priority)
   - `PROJECT_NAME` - explicit project identifier
   - `PROJECT_PATH` - explicit project root directory
   - `CONFIG_FILE` - explicit config file path

2. **Configuration Files** (medium priority)  
   - `pyproject.toml` project.name field
   - `package.json` name field
   - `.graphrag/config.yaml` project settings

3. **Auto-Detection** (lowest priority, fallback only)
   - Git remote origin parsing
   - Directory name sanitization  
   - Project marker files (`.git`, `pyproject.toml`, etc.)

4. **Hard-coded Defaults** (last resort)
   - `DEFAULT_PROJECT_NAME = "default"`

### Implementation Standard

```python
# CORRECT LOGIC (ADR-0037 compliant)
import os
import logging

def load_project_configuration():
    """Load project configuration following ADR-0037 priority hierarchy"""
    
    # 1. PRIORITY: Environment Variables
    project_name = os.getenv("PROJECT_NAME")
    project_path = os.getenv("PROJECT_PATH")
    
    if project_name and project_path:
        if not os.path.isdir(project_path):
            logging.error(f"PROJECT_PATH does not exist: {project_path}")
            sys.exit(1)
        logging.info(f"✅ Using explicit config: PROJECT_NAME='{project_name}', PROJECT_PATH='{project_path}'")
        return project_name, project_path
    
    # 2. FALLBACK: Auto-detection (only if env vars missing)
    if not project_name or not project_path:
        logging.warning("Environment variables missing. Falling back to auto-detection.")
        detected_name, detected_path = auto_detect_project_smart()
        
        project_name = project_name or detected_name
        project_path = project_path or detected_path
        
        logging.info(f"🔍 Auto-detected: PROJECT_NAME='{project_name}', PROJECT_PATH='{project_path}'")
    
    # 3. VALIDATION: Ensure we have valid configuration
    if not project_name or not project_path:
        logging.error("Failed to determine project configuration. Set PROJECT_NAME and PROJECT_PATH explicitly.")
        sys.exit(1)
        
    return project_name, project_path

def auto_detect_project_smart():
    """Improved auto-detection with project marker files"""
    # Look for project markers instead of directory ordering
    for marker in ['.git', 'pyproject.toml', 'package.json', '.graphrag']:
        if os.path.exists(f"/workspace/{marker}"):
            return sanitize_name(os.path.basename("/workspace")), "/workspace"
    
    # Fallback to directory name if no markers found
    return "default", "/workspace"
```

## Consequences

### Positive

✅ **Reliability**: Configuration behavior is predictable and explicit  
✅ **Standards Compliance**: Aligns with Twelve-Factor App methodology  
✅ **Debugging**: Clear configuration source makes troubleshooting easier  
✅ **Scalability**: Standard applies to all containerized services  
✅ **Bug Prevention**: Eliminates entire class of configuration errors

### Negative

⚠️ **Migration Effort**: Existing services may need entrypoint updates  
⚠️ **Documentation**: Teams must understand priority hierarchy  
⚠️ **Validation**: Need automated checks to enforce standard

### Risk Mitigation

1. **Gradual Rollout**: Fix indexer first, then apply to other services
2. **Clear Documentation**: Update CLAUDE.md with configuration requirements  
3. **Code Reviews**: Enforce ADR-0037 compliance in reviews
4. **Automated Testing**: Add configuration validation tests

## Implementation Plan

### Phase 1: Immediate Fix (Critical)
- [ ] Fix indexer container `entrypoint.py` startup logic
- [ ] Test with `PROJECT_NAME=claude-l9-template` and `PROJECT_PATH=/workspace`
- [ ] Verify `project-claude-l9-template` Qdrant collection creation
- [ ] Validate hybrid GraphRAG search functionality

### Phase 2: Standardization (High Priority)
- [ ] Update CLAUDE.md with ADR-0037 requirements
- [ ] Create configuration validation utility
- [ ] Apply standard to other containerized services
- [ ] Add automated tests for configuration priority

### Phase 3: Enforcement (Medium Priority)
- [ ] Code review checklist for ADR-0037 compliance
- [ ] CI/CD validation of configuration standards
- [ ] Documentation and training for team

## Alternatives Considered

### Alternative 1: Ad-hoc Bug Fix
**Approach**: Fix the specific indexer bug without creating standard  
**Rejected**: Short-sighted, leaves system vulnerable to similar issues

### Alternative 2: Config Management Tool  
**Approach**: Use Helm charts or similar for configuration  
**Rejected**: Adds complexity, doesn't solve fundamental priority issue

### Alternative 3: README Documentation
**Approach**: Document configuration in README without formal ADR  
**Rejected**: Lower visibility, less enforcement, not discoverable

## Expert Consensus

### Multi-Model Validation Results

**Gemini-2.5-Pro (FOR)**: 10/10 confidence  
- "Clear-cut best practice with high-leverage impact"
- "Essential for scaling containerized services"

**Grok-4 (NEUTRAL)**: 8/10 confidence  
- "Standard industry practice, lightweight implementation"
- "Reduces maintenance burden and technical debt"

**Industry Alignment**: Twelve-Factor App, Kubernetes, AWS, Spotify practices

## Related ADRs

- **ADR-0029**: Neo4j Multi-Project Isolation (project scope requirements)
- **ADR-0036**: Neo4j Primitive Property Flattening (data compatibility)
- **ADR-0031**: Canonical Knowledge Management (configuration metadata)

## Success Metrics

1. **Immediate**: Indexer creates correct `project-claude-l9-template` collection
2. **Short-term**: Zero configuration-related container startup failures  
3. **Long-term**: All containerized services follow ADR-0037 standard
4. **Quality**: Reduced debugging time for configuration issues

## References

- [Twelve-Factor App - Config](https://12factor.net/config)
- [Container Configuration Best Practices](https://docs.docker.com/config/)
- [Kubernetes Environment Variables](https://kubernetes.io/docs/concepts/configuration/)
- [ThoughtWorks ADR Template](https://github.com/joelparkerhenderson/architecture_decision_record)

---

**Author**: L9 Engineering Team  
**Reviewers**: Multi-Model Expert Consensus (Gemini-2.5-Pro, Grok-4)  
**Next Review Date**: October 12, 2025