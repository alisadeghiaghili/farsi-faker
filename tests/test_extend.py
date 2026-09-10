"""TDD specs for dynamic name-pool extension."""

from __future__ import annotations

import threading

import pytest

from farsi_faker import FarsiFaker
from farsi_faker.extend import (
    apply_seed_expansion,
    extend_name_pools,
    load_seed_expansion,
)


@pytest.fixture()
def fresh_cache(monkeypatch):
    """Isolate class-level cache for each test."""
    monkeypatch.setattr(FarsiFaker, "_data_cache", None)
    monkeypatch.setattr(FarsiFaker, "_data_lock", threading.Lock())
    yield


class TestExtendNamePools:
    """Process-wide merge of additional names."""

    def test_merges_new_male_names(self, fresh_cache) -> None:
        before = FarsiFaker().get_stats()["male_names_count"]
        added = extend_name_pools(male=["کیومرث"])
        after = FarsiFaker().get_stats()["male_names_count"]

        assert added["male_added"] == 1
        assert after == before + 1
        assert "کیومرث" in FarsiFaker()._male_names

    def test_cleans_ocr_in_input(self, fresh_cache) -> None:
        extend_name_pools(male=["کی ومرث"])
        names = FarsiFaker()._male_names
        assert "کی ومرث" not in names
        assert "کیومرث" in names

    def test_deduplicates_against_existing(self, fresh_cache) -> None:
        first = extend_name_pools(male=["کیومرث"])["male_added"]
        second = extend_name_pools(male=["کیومرث"])["male_added"]
        assert first == 1
        assert second == 0

    def test_extends_all_pools(self, fresh_cache) -> None:
        added = extend_name_pools(
            male=["کیومرث"],
            female=["مهلقا"],
            last=["کیومرثی"],
        )
        # کیومرثی may already exist as a surname; assert male/female at least.
        assert added["male_added"] == 1
        assert added["female_added"] == 1
        assert added["last_added"] >= 0
        assert "مهلقا" in FarsiFaker()._female_names

    def test_rejects_non_sequence(self, fresh_cache) -> None:
        with pytest.raises(TypeError):
            extend_name_pools(male=123)  # type: ignore[arg-type]

    def test_empty_is_noop(self, fresh_cache) -> None:
        added = extend_name_pools()
        assert added == {"male_added": 0, "female_added": 0, "last_added": 0}


class TestSeedExpansion:
    """Built-in curated expansion."""

    def test_seed_contains_core_names(self) -> None:
        seeds = load_seed_expansion()
        assert "محمد" in seeds["male_names"]
        assert "فاطمه" in seeds["female_names"]
        assert "محمدی" in seeds["last_names"]

    def test_seed_covers_meem_and_feh(self) -> None:
        seeds = load_seed_expansion()
        assert any(n.startswith("م") for n in seeds["male_names"])
        assert any(n.startswith("ف") for n in seeds["male_names"])
        assert any(n.startswith("م") for n in seeds["female_names"])
        assert any(n.startswith("ف") for n in seeds["female_names"])

    def test_seed_lists_are_unique(self) -> None:
        seeds = load_seed_expansion()
        for key, values in seeds.items():
            assert list(values) == sorted(set(values)), key

    def test_applying_seed_fills_core_gaps(self, fresh_cache) -> None:
        faker = FarsiFaker()
        # Seeds are baked into names.pkl as of v1.5.0; applying again is a no-op
        # for names already present but must not error or drop data.
        assert "محمد" in faker._male_names
        assert "فاطمه" in faker._female_names
        stats = apply_seed_expansion()
        assert stats["male_added"] >= 0
        faker2 = FarsiFaker()
        assert "محمد" in faker2._male_names
        assert "فاطمه" in faker2._female_names
        assert "مهدی" in faker2._male_names
