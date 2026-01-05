# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

No unreleased changes yet.

---

## [1.0.0] - 2025-12-21

### Initial Release

First stable release of Farsi-Faker - A Persian/Farsi fake name generator for Python.

### Added

#### Core Features
- **FarsiFaker class** - Main class for generating Persian names
- **10,000+ authentic names** - Curated from real Iranian datasets
- **Gender-specific generation** - Separate male and female name lists
- **Multiple gender input formats** - Supports English and Persian inputs
  - English: `male`, `female`, `m`, `f`
  - Persian: `مرد`, `زن`, `پسر`, `دختر`, `مذکر`, `مونث`

#### Methods
- `male_first_name()` - Generate male first name
- `female_first_name()` - Generate female first name  
- `first_name(gender)` - Generate first name with optional gender
- `last_name()` - Generate family name
- `full_name(gender)` - Generate complete person with metadata
- `generate_names(count, gender)` - Generate multiple names
- `generate_dataset(count, male_ratio)` - Generate balanced dataset
- `get_stats()` - Get database statistics

#### Convenience Features
- `generate_fake_name()` - Quick one-off name generation function
- **Reproducible results** - Seed support for consistent output
- **Thread-safe** - Safe for concurrent use
- **Class-level caching** - Shared data across instances for memory efficiency

#### Data Management
- **Optimized pickle storage** - Fast loading with minimal footprint
- **Embedded data** - No external files needed at runtime
- **Unicode support** - Full Persian/Farsi character support

#### Developer Experience
- **Zero dependencies** - No external packages required for production
- **Full type hints** - Complete typing.Literal and Optional annotations
- **Comprehensive docstrings** - Detailed documentation for all methods
- **Rich examples** - Code examples in all docstrings

### Testing
- **35+ test cases** - Comprehensive test coverage
- **Unit tests** - Testing individual methods
- **Integration tests** - Real-world usage scenarios
- **Error handling tests** - Edge cases and invalid inputs
- **Performance tests** - Benchmarks for large datasets
- **pytest fixtures** - Reusable test components

### Package Structure
- **Proper package layout** - Following Python best practices
- **Metadata management** - Version info in separate module
- **Build configuration** - setup.py and pyproject.toml
- **Distribution files** - MANIFEST.in for package data

### Documentation
- **Complete README** - Installation, usage, and examples
- **API documentation** - Detailed method documentation
- **Code examples** - Real-world use cases
- **Contributing guidelines** - How to contribute to the project

### Development Tools
- **create_pickle.py script** - Build pickle from CSV sources
- **Development dependencies** - pytest, black, isort, mypy
- **Type checking** - mypy configuration
- **Code formatting** - Black and isort setup

### Configuration Files
- `setup.py` - Package configuration
- `pyproject.toml` - Build system requirements
- `MANIFEST.in` - Distribution file specification
- `.gitignore` - Git ignore patterns

### License
- **MIT License** - Permissive open-source license
- **Full attribution requirements** - Copyright notice required

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (1.x.x) - Incompatible API changes
- **MINOR** version (x.1.x) - New functionality (backwards-compatible)
- **PATCH** version (x.x.1) - Bug fixes (backwards-compatible)

---

## Future Roadmap

### Planned for 1.1.0
- [ ] Add phone number generation
- [ ] Add address generation (city, street)
- [ ] Add email generation
- [ ] Add national ID (کد ملی) generation
- [ ] Add postal code generation

---

## Links

- **GitHub:** https://github.com/alisadeghiaghili/farsi-faker
- **PyPI:** https://pypi.org/project/farsi-faker/
- **Issues:** https://github.com/alisadeghiaghili/farsi-faker/issues
- **Changelog:** https://github.com/alisadeghiaghili/farsi-faker/releases

---

[Unreleased]: https://github.com/alisadeghiaghili/farsi-faker/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/alisadeghiaghili/farsi-faker/releases/tag/v1.0.0
