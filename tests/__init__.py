"""Test suite for farsi-faker package.

This package contains comprehensive tests for the FarsiFaker class
and related functionality.

Test Modules:
    - test_faker: Main test suite for FarsiFaker class
    
Test Categories:
    - Unit tests: Individual component testing
    - Integration tests: Real-world usage scenarios
    - Error handling tests: Edge cases and error conditions
    - Performance tests: Speed and memory efficiency

Usage:
    # Run all tests
    pytest tests/
    
    # Run with coverage
    pytest tests/ --cov=farsi_faker --cov-report=html
    
    # Run specific test file
    pytest tests/test_faker.py -v
    
    # Run specific test class
    pytest tests/test_faker.py::TestFarsiFaker -v
    
    # Run specific test
    pytest tests/test_faker.py::TestFarsiFaker::test_reproducibility -v
"""

__version__ = "1.0.0"
__all__ = []
