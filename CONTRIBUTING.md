# Contributing to Farsi-Faker

Thank you for considering contributing to Farsi-Faker! 🎉

We welcome contributions from everyone. This document provides guidelines for contributing to the project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)

---

## 📜 Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code:

- **Be respectful** - Treat everyone with respect and kindness
- **Be inclusive** - Welcome diverse perspectives and experiences
- **Be constructive** - Focus on what is best for the community
- **Be patient** - Understand that people have different skill levels

---

## 🤝 How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:

- **Clear title** - Describe the issue concisely
- **Steps to reproduce** - How can we reproduce the bug?
- **Expected behavior** - What should happen?
- **Actual behavior** - What actually happens?
- **Environment** - Python version, OS, package version
- **Code sample** - Minimal code that reproduces the issue

**Example:**
```
Title: generate_dataset raises error with male_ratio=1.0

Steps to reproduce:
1. Create faker: faker = FarsiFaker()
2. Call: dataset = faker.generate_dataset(10, male_ratio=1.0)

Expected: Returns 10 male names
Actual: Raises ValueError

Environment:
- Python 3.9
- farsi-faker 1.0.0
- Ubuntu 20.04
```

### Suggesting Enhancements

Feature requests are welcome! Please open an issue with:

- **Clear description** - What feature do you want?
- **Use case** - Why is this feature useful?
- **Examples** - Show how it would work
- **Alternatives** - What alternatives have you considered?

### Pull Requests

We actively welcome pull requests for:

- **Bug fixes** - Fix reported issues
- **New features** - Add new functionality
- **Documentation** - Improve or add documentation
- **Tests** - Add or improve test coverage
- **Performance** - Optimize existing code

---

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/farsi-faker.git
cd farsi-faker
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
# Install package in editable mode with all dependencies
pip install -e ".[all]"

# Or separately:
pip install -e .
pip install pytest pytest-cov black isort mypy pandas
```

### 4. Create a Branch

```bash
# Create and checkout a new branch
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b fix/bug-description
```

---

## 🔄 Pull Request Process

### 1. Make Your Changes

- Write clean, readable code
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation if needed

### 2. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=farsi_faker --cov-report=html

# Check coverage report
open htmlcov/index.html
```

### 3. Format Code

```bash
# Format with Black
black farsi_faker/ tests/

# Sort imports
isort farsi_faker/ tests/

# Type check
mypy farsi_faker/
```

### 4. Commit Changes

```bash
# Stage your changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: description of your changes"

# Good commit messages:
# - "Add: phone number generation method"
# - "Fix: handle None input in full_name()"
# - "Docs: update README with new examples"
# - "Test: add tests for edge cases"
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Then create a Pull Request on GitHub
```

### 6. PR Requirements

Your PR should:

- ✅ Pass all tests
- ✅ Have good test coverage (>80%)
- ✅ Follow coding standards
- ✅ Include documentation updates
- ✅ Have a clear description
- ✅ Reference related issues (if any)

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
Describe how you tested your changes

## Checklist
- [ ] Tests pass
- [ ] Code formatted (black + isort)
- [ ] Type hints added
- [ ] Documentation updated
- [ ] CHANGELOG updated (for features/fixes)
```

---

## 📝 Coding Standards

### Python Style

- **Follow PEP 8** - Python style guide
- **Use Black** - Code formatter (line length: 100)
- **Use isort** - Import sorter
- **Use type hints** - For all functions and methods
- **Write docstrings** - For all public APIs

### Code Example

```python
from typing import Optional, List, Dict


def generate_names(
    count: int = 10,
    gender: Optional[str] = None
) -> List[Dict[str, str]]:
    """Generate multiple full names.

    Args:
        count: Number of names to generate (must be positive).
        gender: Desired gender for all names. If None, randomly mixes genders.

    Returns:
        List of person dictionaries.

    Raises:
        ValueError: If count is not positive.

    Example:
        >>> faker = FarsiFaker()
        >>> names = faker.generate_names(5, 'male')
        >>> len(names)
        5
    """
    if count <= 0:
        raise ValueError(f"Count must be positive, got: {count}")

    return [self.full_name(gender) for _ in range(count)]
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `FarsiFaker`)
- **Functions/Methods**: `snake_case` (e.g., `full_name`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `_GENDER_MAP`)
- **Private**: Prefix with `_` (e.g., `_load_data`)

---

## 🧪 Testing Guidelines

### Test Structure

```python
import pytest
from farsi_faker import FarsiFaker


class TestFeatureName:
    """Test suite for feature."""

    @pytest.fixture
    def faker(self):
        """Create faker instance for tests."""
        return FarsiFaker(seed=42)

    def test_basic_functionality(self, faker):
        """Test basic functionality."""
        result = faker.full_name()
        assert isinstance(result, dict)
        assert 'name' in result

    def test_edge_case(self, faker):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            faker.generate_names(-1)
```

### Test Requirements

- ✅ Test happy path (normal usage)
- ✅ Test edge cases (boundary conditions)
- ✅ Test error cases (invalid inputs)
- ✅ Use fixtures for reusable components
- ✅ Use parametrize for multiple inputs
- ✅ Add docstrings to test functions

---

## 📚 Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def method_name(param1: str, param2: int = 10) -> Dict[str, str]:
    """One-line summary of method.

    Longer description if needed. Explain what the method does,
    any important behavior, or caveats.

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
        TypeError: When wrong type is provided

    Example:
        >>> faker = FarsiFaker()
        >>> result = faker.method_name('test', 20)
        >>> print(result)
        {'key': 'value'}
    """
    pass
```

### README Updates

If adding new features, update:

- [ ] Quick Start section
- [ ] Documentation section
- [ ] Examples section
- [ ] Add to Table of Contents

---

## 🔖 Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking changes
- **MINOR** (1.0.0 → 1.1.0): New features (backwards-compatible)
- **PATCH** (1.0.0 → 1.0.1): Bug fixes (backwards-compatible)

---

## ❓ Questions?

If you have questions:

- **Open an issue** on GitHub
- **Email**: alisadeghiaghili@gmail.com
- **Check existing issues** for similar questions

---

## 🎉 Thank You!

Thank you for contributing to Farsi-Faker! Your efforts help make this project better for everyone.

---

Made with ❤️ by [Ali Sadeghi Aghili](https://github.com/alisadeghiaghili)
