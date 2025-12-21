"""Farsi Faker - Generate realistic fake Persian/Farsi data.

فارسی فیکر - تولید داده‌های فیک فارسی برای تست

This package provides a simple and efficient way to generate realistic Persian/Farsi
names for testing, data generation, mock data, and other applications.

Features:
    - 10,000+ authentic Persian names from real Iranian datasets
    - Gender-specific name generation (male/female)
    - High-performance with pickle-based optimized data storage
    - Thread-safe implementation
    - Reproducible results with seed support
    - Zero external dependencies for production use
    - Fully typed with comprehensive type hints
    - Extensive test coverage

Quick Start:
    >>> from farsi_faker import FarsiFaker
    >>> 
    >>> # Create faker instance
    >>> faker = FarsiFaker()
    >>> 
    >>> # Generate a random male name
    >>> person = faker.full_name('male')
    >>> print(person['name'])
    علی احمدی
    >>> 
    >>> # Generate 100 people with 60% male ratio
    >>> dataset = faker.generate_dataset(100, male_ratio=0.6)
    >>> 
    >>> # Get database statistics
    >>> stats = faker.get_stats()
    >>> print(f"Total combinations: {stats['possible_combinations']:,}")

Examples:
    Generate single names:
        >>> faker = FarsiFaker()
        >>> male_name = faker.male_first_name()
        >>> female_name = faker.female_first_name()
        >>> last_name = faker.last_name()
    
    Generate with reproducible results:
        >>> faker = FarsiFaker(seed=42)
        >>> name1 = faker.full_name()
        >>> faker = FarsiFaker(seed=42)
        >>> name2 = faker.full_name()
        >>> assert name1 == name2  # Same results!
    
    Quick one-off generation:
        >>> from farsi_faker import generate_fake_name
        >>> person = generate_fake_name('female', seed=123)
        >>> print(person['name'])

For detailed documentation, visit: https://github.com/alisadeghiaghili/farsi-faker
"""

from .faker import FarsiFaker, generate_fake_name
from ._version import (
    __version__,
    __version_info__,
    __status__,
    __release_date__,
)

# Public API
__all__ = [
    'FarsiFaker',
    'generate_fake_name',
    '__version__',
    '__version_info__',
]

# Package metadata
__author__ = 'Ali Sadeghi Aghili'
__author_email__ = 'alisadeghiaghili@gmail.com'
__maintainer__ = 'Ali Sadeghi Aghili'
__maintainer_email__ = 'alisadeghiaghili@gmail.com'
__license__ = 'MIT'
__copyright__ = f'Copyright (c) 2025 {__author__}'
__url__ = 'https://github.com/alisadeghiaghili/farsi-faker'
__docs_url__ = 'https://github.com/alisadeghiaghili/farsi-faker#readme'
__source_url__ = 'https://github.com/alisadeghiaghili/farsi-faker'
__tracker_url__ = 'https://github.com/alisadeghiaghili/farsi-faker/issues'
__pypi_url__ = 'https://pypi.org/project/farsi-faker/'
__description__ = 'Generate realistic fake Persian/Farsi names for testing and development'
__long_description__ = __doc__

# Development status
__status__ = __status__  # From _version.py

# Package info for introspection
def get_info():
    """Get package information.
    
    Returns:
        dict: Package metadata including version, author, license, etc.
    
    Example:
        >>> from farsi_faker import get_info
        >>> info = get_info()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Author: {info['author']}")
    """
    return {
        'name': 'farsi-faker',
        'version': __version__,
        'version_info': __version_info__,
        'status': __status__,
        'release_date': __release_date__,
        'author': __author__,
        'author_email': __author_email__,
        'license': __license__,
        'url': __url__,
        'docs_url': __docs_url__,
        'pypi_url': __pypi_url__,
        'description': __description__,
    }


def show_info():
    """Print package information in a formatted way.
    
    Example:
        >>> from farsi_faker import show_info
        >>> show_info()
        farsi-faker v1.0.0
        ==================
        Author: Ali Sadeghi Aghili
        License: MIT
        URL: https://github.com/alisadeghiaghili/farsi-faker
    """
    info = get_info()
    print(f"{info['name']} v{info['version']}")
    print("=" * (len(info['name']) + len(info['version']) + 3))
    print(f"Status: {info['status']}")
    print(f"Release Date: {info['release_date']}")
    print(f"Author: {info['author']} <{info['author_email']}>")
    print(f"License: {info['license']}")
    print(f"Homepage: {info['url']}")
    print(f"PyPI: {info['pypi_url']}")
    print(f"\nDescription: {info['description']}")


# Version check helper
def check_version(required_version: str) -> bool:
    """Check if current version meets the required version.
    
    Args:
        required_version: Minimum required version (e.g., "1.0.0")
    
    Returns:
        bool: True if current version >= required version
    
    Example:
        >>> from farsi_faker import check_version
        >>> if check_version("1.0.0"):
        ...     print("Version OK!")
    """
    try:
        required = tuple(int(i) for i in required_version.split('.') if i.isdigit())
        return __version_info__ >= required
    except (ValueError, AttributeError):
        return False


# Expose get_info and show_info in __all__ if you want them public
__all__.extend(['get_info', 'show_info', 'check_version'])


# Optional: Print info when module is imported with python -m
if __name__ == '__main__':
    show_info()
