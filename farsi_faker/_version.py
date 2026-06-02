"""Version information for farsi-faker package.

This module contains version information following Semantic Versioning 2.0.0.
https://semver.org/

Version Format: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Semantic Versioning Rules:
    - MAJOR: Incompatible API changes (breaking changes)
    - MINOR: Backwards-compatible functionality additions (new features)
    - PATCH: Backwards-compatible bug fixes

Examples:
    - 1.0.0: First stable release
    - 1.0.1: Bug fix release
    - 1.1.0: New feature release (backwards compatible)
    - 2.0.0: Major release with breaking changes

Attributes:
    __version__ (str): The current version string (e.g., "1.1.0")
    __version_info__ (tuple): Version as a tuple (e.g., (1, 1, 0))
    VERSION_MAJOR (int): Major version number
    VERSION_MINOR (int): Minor version number
    VERSION_PATCH (int): Patch version number
    __status__ (str): Development status descriptor
    __release_date__ (str): Release date in ISO format (YYYY-MM-DD)
"""

__version__ = "1.1.0"
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())

# Version components for programmatic access
VERSION_MAJOR = __version_info__[0] if len(__version_info__) > 0 else 0
VERSION_MINOR = __version_info__[1] if len(__version_info__) > 1 else 0
VERSION_PATCH = __version_info__[2] if len(__version_info__) > 2 else 0

# Development status
# Options: "Planning", "Pre-Alpha", "Alpha", "Beta", "Production/Stable", "Mature", "Inactive"
__status__ = "Production/Stable"

# Release information
__release_date__ = "2026-06-02"
__release_name__ = "DataFrame Integration"

# Package metadata
__author__ = "Ali Sadeghi Aghili"
__author_email__ = "alisadeghiaghili@gmail.com"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2025-2026 {__author__}"

# URLs
__url__ = "https://github.com/alisadeghiaghili/farsi-faker"
__docs_url__ = "https://github.com/alisadeghiaghili/farsi-faker#readme"
__issues_url__ = "https://github.com/alisadeghiaghili/farsi-faker/issues"
__pypi_url__ = "https://pypi.org/project/farsi-faker/"


def get_version() -> str:
    """Get the current version string.

    Returns:
        str: Version string (e.g., ``"1.1.0"``)

    Example::

        >>> from farsi_faker._version import get_version
        >>> get_version()
        '1.1.0'
    """
    return __version__


def get_version_info() -> tuple:
    """Get the current version as a tuple.

    Returns:
        tuple: Version tuple (e.g., ``(1, 1, 0)``)

    Example::

        >>> from farsi_faker._version import get_version_info
        >>> get_version_info()
        (1, 1, 0)
    """
    return __version_info__


def get_full_version() -> str:
    """Get a human-readable version string with status and release date.

    Returns:
        str: Formatted version string.

    Example::

        >>> from farsi_faker._version import get_full_version
        >>> get_full_version()
        'farsi-faker v1.1.0 (Production/Stable) - Released: 2026-06-02'
    """
    return (
        f"farsi-faker v{__version__} "
        f"({__status__}) - "
        f"Released: {__release_date__}"
    )


def check_version(required_version: str) -> bool:
    """Check if the installed version meets a minimum requirement.

    Args:
        required_version (str): Minimum required version string
            (e.g., ``"1.1.0"``).

    Returns:
        bool: ``True`` if the current version is greater than or equal to
        *required_version*, ``False`` otherwise.

    Example::

        >>> from farsi_faker._version import check_version
        >>> check_version("1.0.0")
        True
        >>> check_version("2.0.0")
        False
    """
    try:
        required = tuple(int(i) for i in required_version.split('.') if i.isdigit())
        return __version_info__ >= required
    except (ValueError, AttributeError):
        return False


VERSION_HISTORY = {
    "1.1.0": {
        "date": "2026-06-02",
        "status": "stable",
        "changes": [
            "Add as_dataframe parameter to generate_names() and generate_dataset()",
            "Improve male_ratio validation error message with computed counts",
            "Bullet-proof all docstrings with runnable examples and output",
            "Use TYPE_CHECKING guard for pandas import (zero runtime cost)",
            "Fix exception chaining (raise ... from exc) throughout",
            "Add 18 new tests in TestDataFrame covering shape, dtypes, nulls, "
            "gender ratio accuracy, backward compatibility, and pandas workflow",
        ],
    },
    "1.0.0": {
        "date": "2025-12-21",
        "status": "stable",
        "changes": [
            "Initial release with 10,000+ authentic Persian names",
            "Gender-specific name generation (male/female)",
            "High-performance pickle-based data storage",
            "Thread-safe implementation",
            "Reproducible results with seed support",
            "Zero external dependencies",
            "Full type hints support",
            "Comprehensive test coverage",
        ],
    },
}


def get_changelog(version: str = None) -> dict:
    """Get the changelog for a specific version or all versions.

    Args:
        version (str, optional): Version string to look up
            (e.g., ``"1.1.0"``).  When ``None`` (default) the full
            history dict is returned.

    Returns:
        dict: Changelog entry for the requested version, or the complete
        ``VERSION_HISTORY`` dict when *version* is ``None``.
        Returns an empty dict if the requested version is not found.

    Example::

        >>> from farsi_faker._version import get_changelog
        >>> entry = get_changelog("1.1.0")
        >>> entry["date"]
        '2026-06-02'
        >>> isinstance(entry["changes"], list)
        True
    """
    if version:
        return VERSION_HISTORY.get(version, {})
    return VERSION_HISTORY


def is_stable() -> bool:
    """Return ``True`` if the current release is marked as stable.

    Returns:
        bool: ``True`` when ``__status__ == 'Production/Stable'``.
    """
    return __status__ == "Production/Stable"


def is_development() -> bool:
    """Return ``True`` if the current release is a pre-release.

    Returns:
        bool: ``True`` when the status is one of
        ``'Planning'``, ``'Pre-Alpha'``, ``'Alpha'``, or ``'Beta'``.
    """
    return __status__ in ["Planning", "Pre-Alpha", "Alpha", "Beta"]


__all__ = [
    '__version__',
    '__version_info__',
    'VERSION_MAJOR',
    'VERSION_MINOR',
    'VERSION_PATCH',
    '__status__',
    '__release_date__',
    '__author__',
    '__license__',
    'get_version',
    'get_version_info',
    'get_full_version',
    'check_version',
    'get_changelog',
    'is_stable',
    'is_development',
]


if __name__ == '__main__':
    print("=" * 70)
    print(get_full_version())
    print("=" * 70)
    print(f"Version String: {__version__}")
    print(f"Version Tuple:  {__version_info__}")
    print(f"Status:         {__status__}")
    print(f"Release Date:   {__release_date__}")
    print(f"Release Name:   {__release_name__}")
    print(f"Author:         {__author__}")
    print(f"License:        {__license__}")
    print(f"URL:            {__url__}")
    print("=" * 70)
    print("\nChangelog:")
    for ver, info in VERSION_HISTORY.items():
        print(f"\nVersion {ver} ({info['date']}):")
        for change in info['changes']:
            print(f"  - {change}")
    print("=" * 70)
