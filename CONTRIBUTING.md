# Contributing to L9 GraphRAG

Thank you for your interest in contributing to L9 GraphRAG! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/l9-graphrag.git
   cd l9-graphrag
   ```

2. **Set up Development Environment**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   pytest
   ```

## 📋 Development Guidelines

### Code Style

We follow industry-standard Python conventions:

- **Formatting**: Black (line length 88)
- **Import Sorting**: isort with Black profile
- **Linting**: Flake8 with Black compatibility
- **Type Checking**: MyPy with strict mode
- **Security**: Bandit for security scanning

All code is automatically formatted and checked via pre-commit hooks.

### Git Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Locally**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test categories
   pytest tests/unit/
   pytest tests/integration/
   
   # Check coverage
   pytest --cov=src --cov-report=html
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add hybrid search optimization"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Build process or auxiliary tool changes

**Examples:**
```
feat(graphrag): add impact analysis tool
fix(indexer): resolve file debouncing race condition
docs: update installation guide
test: add hybrid retriever unit tests
```

## 🏗 Architecture Guidelines

### Project Structure

```
src/
├── graphrag/           # Core GraphRAG functionality
│   ├── core/          # Core algorithms and logic
│   ├── services/      # Service layer (indexing, retrieval)
│   ├── models/        # Data models and schemas
│   └── utils/         # Shared utilities
└── mcp/               # MCP server implementation
```

### Design Principles

1. **Separation of Concerns**: Clear boundaries between components
2. **Dependency Injection**: Services should be injectable and testable
3. **Async First**: Use async/await for I/O operations
4. **Error Handling**: Comprehensive error handling and logging
5. **Configuration**: Environment-based configuration
6. **Testing**: High test coverage with unit and integration tests

### Adding New Features

1. **Core Logic**: Add to appropriate `src/graphrag/` subdirectory
2. **Tests**: Add comprehensive tests in `tests/`
3. **Documentation**: Update relevant docs in `docs/`
4. **Configuration**: Add config options to `.env.example`
5. **Dependencies**: Update `requirements/base.txt` if needed

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Fast, isolated unit tests
├── integration/       # Integration tests with services
├── fixtures/          # Shared test data and utilities
└── performance/       # Performance and load tests
```

### Test Categories

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical paths
- **End-to-End Tests**: Full system testing

### Writing Tests

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self):
        # Arrange
        retriever = HybridRetriever(mock_config)
        
        # Act
        results = await retriever.find_similar_with_context("test query")
        
        # Assert
        assert len(results) > 0
        assert results[0].score > 0.5

    @pytest.mark.integration
    async def test_with_real_databases(self):
        # Integration test with actual Neo4j/Qdrant
        pass
```

### Test Data

- Use factories for creating test data
- Mock external services in unit tests
- Use test containers for integration tests

## 📚 Documentation

### Types of Documentation

1. **API Documentation**: Docstrings in code
2. **User Guides**: Step-by-step tutorials
3. **Architecture Docs**: System design documentation
4. **Development Docs**: Setup and contribution guides

### Writing Documentation

- Use clear, concise language
- Include code examples
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

## 🔍 Code Review Process

### Submitting PRs

1. **Clear Description**: Explain what changes and why
2. **Link Issues**: Reference related issues
3. **Screenshots**: Include UI changes if applicable
4. **Tests**: Ensure all tests pass
5. **Documentation**: Update docs as needed

### Review Criteria

- **Functionality**: Does it work as intended?
- **Code Quality**: Is it maintainable and readable?
- **Performance**: Does it introduce performance issues?
- **Security**: Are there any security concerns?
- **Testing**: Is it adequately tested?

## 🐛 Reporting Issues

### Bug Reports

Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
- Relevant logs or error messages

### Feature Requests

Include:
- Clear use case description
- Proposed solution or approach
- Alternative solutions considered
- Impact assessment

## 🏷 Release Process

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions  
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. Publish to PyPI (if applicable)

## 💬 Community

### Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Email**: enterprise@your-org.com for enterprise support

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## 📋 Checklist

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts

Thank you for contributing to L9 GraphRAG! 🎉