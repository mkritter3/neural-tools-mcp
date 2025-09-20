# ADR-0056: Python Import Architecture and sys.path Management

## Status
Accepted (September 19, 2025)

## Context

During CI/CD validation improvements, we discovered that our Python import system relies on runtime `sys.path` manipulations to function. Initial attempts to "clean up" these manipulations and use only proper Python imports resulted in breaking core functionality including:
- Pattern extraction (ADR-0031)
- Code parsing
- PRISM scoring
- Service container imports

Investigation revealed that tests were passing with incorrect imports because they add multiple directories to `sys.path`, making the test environment more permissive than production MCP environment.

## The Problem

Our codebase has evolved with mixed import patterns:

1. **Relative imports**: `from service_container import ServiceContainer`
2. **Absolute imports**: `from servers.services.service_container import ServiceContainer`
3. **Cross-package imports**: Services importing from `infrastructure/` package

The MCP server only adds `neural-tools/src` to `sys.path`, but many modules expect their parent directories to also be in the path.

## Decision

**We will KEEP the sys.path manipulations as necessary technical debt** until a proper refactoring can be done.

This means:
1. **Preserve existing sys.path modifications** in service files
2. **Document why they exist** with clear comments
3. **Create validation that works WITH these modifications**
4. **Plan future refactoring** to eliminate the need for them

## Detailed Analysis

### Why sys.path Manipulations Exist

```python
# In indexer_service.py
services_dir = Path(__file__).parent
sys.path.insert(0, str(services_dir))  # Allows: from service_container import ...
sys.path.insert(0, str(services_dir.parent))  # Allows: from config.collection_naming import ...
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "infrastructure"))  # Allows: from prism_scorer import ...
```

These enable:
- Services to import from each other without full paths
- Cross-package imports between `servers/` and `infrastructure/`
- Configuration imports from parent directories

### Why Removing Them Breaks Everything

Without these modifications:
- `from pattern_extractor import PatternExtractor` fails (module not found)
- `from code_parser import CodeParser` fails (module not found)
- `from prism_scorer import PrismScorer` fails (module not found)
- `from service_container import ServiceContainer` fails (module not found)

These are CORE functionality, not optional features.

## Implementation Plan

### Phase 1: Documentation and Protection (Immediate)

1. **Add warning comments** to all sys.path modifications:
```python
# CRITICAL: DO NOT REMOVE - Required for imports to work (see ADR-0056)
# This compensates for mixed import patterns across the codebase
sys.path.insert(0, str(services_dir))
```

2. **Update CI/CD validation** to test WITH sys.path modifications:
```python
# ci/validate_environment.py should simulate actual import behavior
# including the sys.path modifications that services make
```

3. **Document the import patterns** each service expects

### Phase 2: Validation that Reflects Reality (This Week)

1. **Create realistic import tests** that include sys.path modifications
2. **Test both import styles** work as expected
3. **Validate critical services** can import their dependencies

### Phase 3: Future Refactoring (Technical Debt Item)

Long-term solution requires:
1. **Standardize on absolute imports** everywhere
2. **Break circular dependencies** between services
3. **Use proper package structure** with `__init__.py` files
4. **Eliminate cross-package imports** or use proper entry points

## Consequences

### Positive
- **System remains functional** - No broken production deployments
- **Clear documentation** of technical debt
- **Realistic validation** that matches production behavior
- **Gradual improvement path** without breaking changes

### Negative
- **Technical debt remains** - sys.path hacks are not ideal
- **Import confusion** - Mixed patterns make onboarding harder
- **Testing complexity** - Tests must account for runtime path modifications
- **Potential for errors** - Easy to break imports without realizing

## Alternatives Considered

1. **Remove all sys.path modifications** - REJECTED: Breaks core functionality
2. **Refactor everything now** - REJECTED: Too risky, would delay critical features
3. **Add more sys.path modifications** - REJECTED: Makes problem worse
4. **Use PYTHONPATH environment variable** - REJECTED: Doesn't solve mixed import patterns

## Metrics for Success

- Zero import-related production failures
- CI/CD catches import issues before deployment
- Clear documentation prevents accidental breakage
- Future refactoring ticket created and prioritized

## References

- [PEP 328 - Imports: Multi-Line and Absolute/Relative](https://www.python.org/dev/peps/pep-0328/)
- [Python Import System Documentation](https://docs.python.org/3/reference/import.html)
- Gemini's deep analysis of validation blind spots (September 19, 2025)

## Implementation Checklist

- [ ] Add warning comments to all sys.path modifications
- [ ] Update validate_environment.py to work with current architecture
- [ ] Create import pattern documentation
- [ ] Add import validation to pre-commit hooks
- [ ] Create technical debt ticket for future refactoring
- [ ] Update developer onboarding docs

## Notes

This is pragmatic engineering - acknowledging that perfect is the enemy of good. The sys.path modifications are not ideal, but they work and removing them without proper refactoring breaks production. We choose working code with documented technical debt over broken "clean" code.