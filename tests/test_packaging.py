"""Packaging and package-integrity tests for farsi-faker."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import farsi_faker
from farsi_faker import FarsiFaker


PACKAGE_ROOT = Path(farsi_faker.__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class TestPackageIntegrity:
    """Source-tree integrity checks that protect the install experience."""

    def test_py_typed_marker_exists(self) -> None:
        """PEP 561 marker must ship so type checkers see the inline types."""
        assert (PACKAGE_ROOT / "py.typed").is_file()

    def test_names_pickle_exists(self) -> None:
        """Embedded names database must be present next to the package code."""
        assert (PACKAGE_ROOT / "data" / "names.pkl").is_file()

    def test_requires_python_is_39_plus(self) -> None:
        """Declared Python floor must match typing constructs used in source."""
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'requires-python = ">=3.9"' in pyproject

    def test_create_pickle_module_does_not_autorun(self) -> None:
        """Importing the data-build script must not execute main() or exit."""
        script = PROJECT_ROOT / "scripts" / "create_pickle.py"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import runpy; "
                    "ns = runpy.run_path(r'%s', run_name='not_main'); "
                    "assert 'main' in ns; "
                    "print('IMPORT_OK')"
                )
                % str(script),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "IMPORT_OK" in result.stdout

    def test_data_cache_is_tuple_after_load(self) -> None:
        """Shared class-level cache must expose immutable name sequences."""
        FarsiFaker()
        cache = FarsiFaker._data_cache
        assert cache is not None
        assert isinstance(cache["male_names"], tuple)
        assert isinstance(cache["female_names"], tuple)
        assert isinstance(cache["last_names"], tuple)


class TestVersionSurface:
    """Version helpers must stay consistent for release automation."""

    def test_version_is_semver_triple(self) -> None:
        parts = farsi_faker.__version_info__
        assert len(parts) == 3
        assert all(isinstance(part, int) and part >= 0 for part in parts)

    def test_public_check_version_matches_private_impl(self) -> None:
        """Single source of truth for version comparison."""
        from farsi_faker import _version

        assert farsi_faker.check_version is _version.check_version

    def test_check_version_contract(self) -> None:
        assert farsi_faker.check_version(farsi_faker.__version__) is True
        assert farsi_faker.check_version("99.0.0") is False
        assert farsi_faker.check_version("not-a-version") is False
