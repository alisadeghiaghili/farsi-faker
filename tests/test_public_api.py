"""Tests for public package helpers and the data package guard."""

from __future__ import annotations

import pytest

import farsi_faker
from farsi_faker import _version


class TestPublicHelpers:
    """get_info / show_info / package metadata surface."""

    def test_get_info_keys(self) -> None:
        info = farsi_faker.get_info()
        expected = {
            "name",
            "version",
            "version_info",
            "status",
            "release_date",
            "author",
            "license",
            "url",
            "description",
        }
        assert expected.issubset(info.keys())
        assert info["name"] == "farsi-faker"
        assert info["version"] == farsi_faker.__version__

    def test_show_info_prints_banner(self, capsys: pytest.CaptureFixture) -> None:
        farsi_faker.show_info()
        captured = capsys.readouterr().out
        assert f"farsi-faker v{farsi_faker.__version__}" in captured
        assert "License: MIT" in captured


class TestVersionModule:
    """Version helper coverage."""

    def test_get_full_version_shape(self) -> None:
        text = _version.get_full_version()
        assert text.startswith("farsi-faker v")
        assert "Production/Stable" in text

    def test_is_stable_and_not_development(self) -> None:
        assert _version.is_stable() is True
        assert _version.is_development() is False

    def test_get_changelog_unknown_returns_empty(self) -> None:
        assert _version.get_changelog("0.0.0") == {}
        assert "1.1.1" in _version.get_changelog()

    def test_check_version_rejects_non_string(self) -> None:
        assert _version.check_version(None) is False  # type: ignore[arg-type]
        assert _version.check_version(1.1) is False  # type: ignore[arg-type]

    def test_check_version_pads_missing_components(self) -> None:
        assert _version.check_version("1") is True
        assert _version.check_version("1.1") is True
        assert _version.check_version(f"{_version.__version__}.0") is True


class TestDataPackageGuard:
    """Direct imports from farsi_faker.data must be blocked."""

    def test_direct_attribute_access_raises(self) -> None:
        from farsi_faker import data

        with pytest.raises(AttributeError, match="should not be accessed directly"):
            _ = data.names  # type: ignore[attr-defined]

    def test_all_is_empty(self) -> None:
        from farsi_faker import data

        assert data.__all__ == []
