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
    __version__ (str): The current version string (e.g., "1.1.1")
    __version_info__ (tuple): Version as a tuple (e.g., (1, 1, 1))
    VERSION_MAJOR (int): Major version number
    VERSION_MINOR (int): Minor version number
    VERSION_PATCH (int): Patch version number
    __status__ (str): Development status descriptor
    __release_date__ (str): Release date in ISO format (YYYY-MM-DD)
"""

from typing import Optional

__version__ = "1.5.0"
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())

# Version components for programmatic access
VERSION_MAJOR = __version_info__[0] if len(__version_info__) > 0 else 0
VERSION_MINOR = __version_info__[1] if len(__version_info__) > 1 else 0
VERSION_PATCH = __version_info__[2] if len(__version_info__) > 2 else 0

# Development status
# Options: "Planning", "Pre-Alpha", "Alpha", "Beta", "Production/Stable", "Mature", "Inactive"
__status__ = "Production/Stable"

# Release information
__release_date__ = "2026-09-10"
__release_name__ = "Extend & Core Coverage"

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
        str: Version string (e.g., ``"1.1.1"``)

    Example::

        >>> from farsi_faker._version import get_version
        >>> get_version() == __version__
        True
    """
    return __version__


def get_version_info() -> tuple:
    """Get the current version as a tuple.

    Returns:
        tuple: Version tuple (e.g., ``(1, 1, 1)``)

    Example::

        >>> from farsi_faker._version import get_version_info
        >>> get_version_info() == __version_info__
        True
    """
    return __version_info__


def get_full_version() -> str:
    """Get a human-readable version string with status and release date.

    Returns:
        str: Formatted version string.

    Example::

        >>> from farsi_faker._version import get_full_version
        >>> get_full_version().startswith('farsi-faker v')
        True
    """
    return (
        f"farsi-faker v{__version__} "
        f"({__status__}) - "
        f"Released: {__release_date__}"
    )


def check_version(required_version: str) -> bool:
    """Check if the installed version meets a minimum requirement.

    Args:
        required_version (str): Minimum required dotted version string
            (e.g., ``"1.1.0"``). Every component must be a non-negative
            integer.

    Returns:
        bool: ``True`` if the current version is greater than or equal to
        *required_version*. ``False`` if the requirement is unmet or the
        input is not a valid dotted version.

    Example::

        >>> from farsi_faker._version import check_version
        >>> check_version("1.0.0")
        True
        >>> check_version("2.0.0")
        False
        >>> check_version("not-a-version")
        False
        >>> check_version("")
        False
    """
    if not isinstance(required_version, str):
        return False

    parts = required_version.strip().split('.')
    if not parts or not all(part.isdigit() for part in parts):
        return False

    required = tuple(int(part) for part in parts)
    # Pad the shorter side so (1, 1) vs (1, 1, 0) compares correctly.
    width = max(len(required), len(__version_info__))
    required = required + (0,) * (width - len(required))
    current = __version_info__ + (0,) * (width - len(__version_info__))
    return current >= required


VERSION_HISTORY = {
    "1.5.0": {
        "date": "2026-09-10",
        "status": "stable",
        "changes": [
            "Fill critical coverage gap: core names starting with م and ف",
            "Bake curated seed expansion into names.pkl (محمد، فاطمه، مهدی، …)",
            "male 7493→7546, female 3648→3695, last 5748→5751",
            "Add farsi_faker.extend.extend_name_pools for runtime merges",
            "Add apply_seed_expansion / load_seed_expansion",
            "CI gates for core common names and م/ف initial coverage",
        ],
    },
    "1.4.0": {
        "date": "2026-09-10",
        "status": "stable",
        "changes": [
            "Add DATA_PROVENANCE.md documenting source status and rebuild pipeline",
            "Add normalize_zwnj / apply_zwnj_policy for ZWNJ hygiene",
            "Wire ZWNJ normalization into precision_repair",
            "Add iranian_cities, city_name, street_name, address_record",
            "Extend profile() / profile_record with city, street, alley, plaque",
        ],
    },
    "1.3.0": {
        "date": "2026-09-10",
        "status": "stable",
        "changes": [
            "Precision-repair name pools: glue Abdol family, drop truncated ال and bare prefixes",
            "Rebuild names.pkl: male 7633→7493, female 3730→3648",
            "Add farsi_faker.profile: national_id (checksum), mobile, email, postal_code",
            "Add FarsiFaker.profile() and field helpers on the class",
            "Add CLI: python -m farsi_faker (JSON/CSV, --profile, --seed)",
        ],
    },
    "1.2.0": {
        "date": "2026-09-10",
        "status": "stable",
        "changes": [
            "Add farsi_faker.cleaning with OCR-split repair and gender-label filters",
            "Rebuild names.pkl: remove singleton-token OCR artifacts and honorific noise",
            "Male pool 7863→7633, female 3817→3730; zero singleton tokens and zero gender overlap",
            "Preserve legitimate compounds (محمد رضا) and multi-word surnames (آب روشن)",
            "Add scripts/rebuild_names_pkl.py",
            "Add data-quality CI gates against the embedded database",
        ],
    },
    "1.1.1": {
        "date": "2026-07-24",
        "status": "stable",
        "changes": [
            "Fix create_pickle.py so importing the script no longer runs main()",
            "Align count validation tests with runtime error messages",
            "Replace flaky single-space full_name assertion with join contract",
            "Load name pools under a lock and expose them as immutable tuples",
            "Document concurrent instantiation safety (one instance per thread)",
            "Add py.typed and declare requires-python >=3.9",
            "Consolidate packaging metadata into pyproject.toml",
            "Deduplicate check_version and reject non-semver inputs",
            "Add packaging integrity tests and CI test workflow",
        ],
    },
    "1.1.0": {
        "date": "2026-06-02",
        "status": "stable",
        "changes": [
            "Add as_dataframe parameter to generate_names() and generate_dataset()",
            "Improve male_ratio validation error message with computed counts",
            "Use TYPE_CHECKING guard for pandas import (zero runtime cost)",
            "Fix exception chaining (raise ... from exc) throughout",
            "Add tests for as_dataframe covering shape, dtypes, nulls, and ratios",
        ],
    },
    "1.0.0": {
        "date": "2025-12-21",
        "status": "stable",
        "changes": [
            "Initial release with embedded Persian names database",
            "Gender-specific name generation (male/female)",
            "Pickle-based data storage",
            "Reproducible results with seed support",
            "Zero external dependencies",
            "Full type hints support",
            "Test suite for core generators",
        ],
    },
}


def get_changelog(version: Optional[str] = None) -> dict:
    """Get the changelog for a specific version or all versions.

    Args:
        version (str, optional): Version string to look up
            (e.g., ``"1.1.1"``).  When ``None`` (default) the full
            history dict is returned.

    Returns:
        dict: Changelog entry for the requested version, or the complete
        ``VERSION_HISTORY`` dict when *version* is ``None``.
        Returns an empty dict if the requested version is not found.

    Example::

        >>> from farsi_faker._version import get_changelog
        >>> entry = get_changelog("1.1.1")
        >>> entry["date"]
        '2026-07-24'
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
