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
    - 1.0.0-alpha.1: Pre-release alpha version
    - 1.0.0-beta.2: Pre-release beta version
    - 1.0.0-rc.1: Release candidate
    - 1.0.0+20251221: Build metadata

Attributes:
    __version__ (str): The current version string (e.g., "1.0.0")
    __version_info__ (tuple): Version as a tuple (e.g., (1, 0, 0))
    VERSION_MAJOR (int): Major version number
    VERSION_MINOR (int): Minor version number
    VERSION_PATCH (int): Patch version number
    __status__ (str): Development status descriptor
    __release_date__ (str): Release date in ISO format (YYYY-MM-DD)
"""

__version__ = "1.0.0"
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())

# Version components for programmatic access
VERSION_MAJOR = __version_info__[0] if len(__version_info__) > 0 else 0
VERSION_MINOR = __version_info__[1] if len(__version_info__) > 1 else 0
VERSION_PATCH = __version_info__[2] if len(__version_info__) > 2 else 0

# Development status
# Options: "Planning", "Pre-Alpha", "Alpha", "Beta", "Production/Stable", "Mature", "Inactive"
__status__ = "Production/Stable"

# Release information
__release_date__ = "2025-12-21"
__release_name__ = "Initial Release"

# Package metadata (duplicated from __init__.py for convenience)
__author__ = "Ali Sadeghi Aghili"
__author_email__ = "alisadeghiaghili@gmail.com"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2025 {__author__}"

# URLs
__url__ = "https://github.com/alisadeghiaghili/farsi-faker"
__docs_url__ = "https://github.com/alisadeghiaghili/farsi-faker#readme"
__issues_url__ = "https://github.com/alisadeghiaghili/farsi-faker/issues"
__pypi_url__ = "https://pypi.org/project/farsi-faker/"


def get_version() -> str:
    """Get the current version string.
    
    Returns:
        str: Version string (e.g., "1.0.0")
    
    Example:
        >>> from farsi_faker._version import get_version
        >>> print(get_version())
        1.0.0
    """
    return __version__


def get_version_info() -> tuple:
    """Get the current version as a tuple.
    
    Returns:
        tuple: Version tuple (e.g., (1, 0, 0))
    
    Example:
        >>> from farsi_faker._version import get_version_info
        >>> major, minor, patch = get_version_info()
        >>> print(f"Version: {major}.{minor}.{patch}")
        Version: 1.0.0
    """
    return __version_info__


def get_full_version() -> str:
    """Get detailed version information.
    
    Returns:
        str: Formatted version information with status and date
    
    Example:
        >>> from farsi_faker._version import get_full_version
        >>> print(get_full_version())
        farsi-faker v1.0.0 (Production/Stable) - Released: 2025-12-21
    """
    return (
        f"farsi-faker v{__version__} "
        f"({__status__}) - "
        f"Released: {__release_date__}"
    )


def check_version(required_version: str) -> bool:
    """Check if current version meets the required version.
    
    Args:
        required_version: Minimum required version string (e.g., "1.0.0")
    
    Returns:
        bool: True if current version >= required version
    
    Example:
        >>> from farsi_faker._version import check_version
        >>> if check_version("1.0.0"):
        ...     print("Version OK")
        Version OK
    """
    try:
        required = tuple(int(i) for i in required_version.split('.') if i.isdigit())
        return __version_info__ >= required
    except (ValueError, AttributeError):
        return False


# Version history and changelog (optional but useful)
VERSION_HISTORY = {
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
        ]
    },
    # Future versions will be added here
    # "1.0.1": {
    #     "date": "2025-XX-XX",
    #     "status": "stable",
    #     "changes": [
    #         "Bug fixes...",
    #     ]
    # },
}


def get_changelog(version: str = None) -> dict:
    """Get changelog for a specific version or all versions.
    
    Args:
        version: Specific version to get changelog for. If None, returns all.
    
    Returns:
        dict: Changelog information
    
    Example:
        >>> from farsi_faker._version import get_changelog
        >>> changelog = get_changelog("1.0.0")
        >>> print(changelog["changes"])
    """
    if version:
        return VERSION_HISTORY.get(version, {})
    return VERSION_HISTORY


# Compatibility checks
def is_stable() -> bool:
    """Check if this is a stable release.
    
    Returns:
        bool: True if version is stable (not alpha, beta, or rc)
    """
    return __status__ == "Production/Stable"


def is_development() -> bool:
    """Check if this is a development version.
    
    Returns:
        bool: True if version is in development (alpha, beta, pre-alpha)
    """
    return __status__ in ["Planning", "Pre-Alpha", "Alpha", "Beta"]


# Module-level string representation
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
    # Print version info when module is run directly
    print("=" * 70)
    print(get_full_version())
    print("=" * 70)
    print(f"Version String: {__version__}")
    print(f"Version Tuple:  {__version_info__}")
    print(f"Major:          {VERSION_MAJOR}")
    print(f"Minor:          {VERSION_MINOR}")
    print(f"Patch:          {VERSION_PATCH}")
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
