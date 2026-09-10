"""Farsi Faker - Generate Persian/Farsi names for testing and development.

Generate Persian/Farsi names for tests, fixtures, mock data, and local
development without external service dependencies.

Features:
    - Embedded Persian name database (male, female, family names)
    - Gender-specific name generation
    - Reproducible output via seed
    - Optional pandas DataFrame output
    - Zero required runtime dependencies
    - PEP 561 typed package (``py.typed``)

Quick Start:
    >>> from farsi_faker import FarsiFaker
    >>> faker = FarsiFaker(seed=42)
    >>> person = faker.full_name('male')
    >>> person['gender']
    'male'
    >>> dataset = faker.generate_dataset(100, male_ratio=0.6)
    >>> len(dataset)
    100
    >>> stats = faker.get_stats()
    >>> stats['male_names_count'] > 0
    True

Homepage: https://github.com/alisadeghiaghili/farsi-faker
"""

from ._version import (
    __release_date__,
    __status__,
    __version__,
    __version_info__,
    check_version,
)
from .faker import FarsiFaker, generate_fake_name

__all__ = [
    'FarsiFaker',
    'generate_fake_name',
    '__version__',
    '__version_info__',
    '__status__',
    '__release_date__',
    'check_version',
    'get_info',
    'show_info',
]

__author__ = 'Ali Sadeghi Aghili'
__author_email__ = 'alisadeghiaghili@gmail.com'
__license__ = 'MIT'
__copyright__ = f'Copyright (c) 2025-2026 {__author__}'
__url__ = 'https://github.com/alisadeghiaghili/farsi-faker'
__description__ = 'Generate Persian/Farsi names for testing and development'


def get_info() -> dict:
    """Return package metadata for introspection.

    Returns:
        dict: Keys include ``name``, ``version``, ``version_info``,
        ``status``, ``release_date``, ``author``, ``license``, ``url``,
        and ``description``.

    Example:
        >>> from farsi_faker import get_info
        >>> info = get_info()
        >>> info['name']
        'farsi-faker'
        >>> info['version'] == __import__('farsi_faker').__version__
        True
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
        'description': __description__,
    }


def show_info() -> None:
    """Print package metadata to stdout.

    Example:
        >>> from farsi_faker import show_info
        >>> show_info()  # doctest: +ELLIPSIS
        farsi-faker v...
    """
    info = get_info()
    banner = f"{info['name']} v{info['version']}"
    print(banner)
    print('=' * len(banner))
    print(f"Status: {info['status']}")
    print(f"Release Date: {info['release_date']}")
    print(f"Author: {info['author']} <{info['author_email']}>")
    print(f"License: {info['license']}")
    print(f"Homepage: {info['url']}")
    print(f"\nDescription: {info['description']}")


if __name__ == '__main__':
    show_info()
