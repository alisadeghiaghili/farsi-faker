"""TDD specs for name-pool cleaning rules."""

from __future__ import annotations

import pytest

from farsi_faker.cleaning import (
    clean_name_pools,
    dedupe_preserve_order,
    drop_from_pool,
    has_singleton_token,
    is_too_short,
    is_truncated_name,
    join_abdol_family,
    join_ocr_splits,
    normalize_name,
    precision_repair,
)


class TestNormalizeName:
    """Whitespace / Unicode normalization."""

    def test_strips_and_collapses_spaces(self) -> None:
        assert normalize_name("  علی   احمدی ") == "علی احمدی"

    def test_empty_and_none(self) -> None:
        assert normalize_name("") == ""
        assert normalize_name("   ") == ""
        assert normalize_name(None) == ""  # type: ignore[arg-type]

    def test_preserves_zwnj(self) -> None:
        assert normalize_name("می‌رود") == "می‌رود"


class TestHasSingletonToken:
    """Detect OCR splits that leave a one-character token."""

    def test_true_when_token_is_single_char(self) -> None:
        assert has_singleton_token("آ رمان") is True
        assert has_singleton_token("د ا و د") is True

    def test_false_for_normal_compound(self) -> None:
        assert has_singleton_token("محمد رضا") is False
        assert has_singleton_token("علی") is False

    def test_false_for_two_char_tokens(self) -> None:
        assert has_singleton_token("با با") is False
        assert has_singleton_token("بی بی") is False


class TestJoinOcrSplits:
    """Join corrupted spaced forms back into single tokens."""

    def test_joins_singleton_token_names(self) -> None:
        assert join_ocr_splits("آ رمان") == "آرمان"
        assert join_ocr_splits("ا میر") == "امیر"
        assert join_ocr_splits("ر حما ن") == "رحمان"
        assert join_ocr_splits("د ا و د") == "داود"

    def test_joins_two_token_when_first_token_is_two_chars(self) -> None:
        assert join_ocr_splits("آر مان") == "آرمان"
        assert join_ocr_splits("با با") == "بابا"
        assert join_ocr_splits("بیژ ن") == "بیژن"
        assert join_ocr_splits("جا بر") == "جابر"

    def test_keeps_legitimate_compounds(self) -> None:
        assert join_ocr_splits("محمد رضا") == "محمد رضا"
        assert join_ocr_splits("علی اصغر") == "علی اصغر"
        assert join_ocr_splits("آقا رضا") == "آقا رضا"
        assert join_ocr_splits("سید محمد") == "سید محمد"

    def test_expands_truncated_agha_prefix(self) -> None:
        # Leading آق/اق before another token is usually truncated آقا/اقا.
        assert join_ocr_splits("آق محمد") == "آقا محمد"
        assert join_ocr_splits("اق قلی") == "اقا قلی"

    def test_noop_for_single_token(self) -> None:
        assert join_ocr_splits("علی") == "علی"
        assert join_ocr_splits("احمدی") == "احمدی"

    def test_empty_passthrough(self) -> None:
        assert join_ocr_splits("") == ""


class TestDropFromPool:
    """Gender-label noise filters."""

    def test_drops_female_honorifics_from_male(self) -> None:
        assert drop_from_pool("بی بی مریم", pool_gender="male") is True
        assert drop_from_pool("آق بی بی", pool_gender="male") is True
        assert drop_from_pool("بلورخانم", pool_gender="male") is True
        assert drop_from_pool("آغاخانم", pool_gender="male") is True

    def test_keeps_normal_male_names(self) -> None:
        assert drop_from_pool("علی", pool_gender="male") is False
        assert drop_from_pool("محمد رضا", pool_gender="male") is False

    def test_drops_agha_titles_from_female_when_clear(self) -> None:
        assert drop_from_pool("آقا رضا", pool_gender="female") is True

    def test_keeps_normal_female_names(self) -> None:
        assert drop_from_pool("فاطمه", pool_gender="female") is False
        assert drop_from_pool("زهرا", pool_gender="female") is False

    def test_last_names_not_filtered(self) -> None:
        assert drop_from_pool("محمدی", pool_gender="last") is False
        assert drop_from_pool("آب روشن", pool_gender="last") is False


class TestDedupePreserveOrder:
    """Stable deduplication."""

    def test_removes_duplicates_keeps_first(self) -> None:
        assert dedupe_preserve_order(["الف", "ب", "الف", "ج", "ب"]) == ["الف", "ب", "ج"]

    def test_empty(self) -> None:
        assert dedupe_preserve_order([]) == []


class TestCleanNamePools:
    """End-to-end pool rebuild."""

    def _raw(self) -> dict[str, list[str]]:
        return {
            "male_names": [
                "علی",
                "آ رمان",
                "آرمان",
                "آر مان",
                "بی بی مریم",
                "محمد رضا",
                "ا میر",
            ],
            "female_names": [
                "فاطمه",
                "ا ناهید",
                "آناهید",
                "زهرا",
            ],
            "last_names": [
                "احمدی",
                "احمدی",
                "آب روشن",
                "رضایی",
            ],
        }

    def test_joins_and_dedupes_male_pool(self) -> None:
        cleaned = clean_name_pools(self._raw())
        assert "آ رمان" not in cleaned["male_names"]
        assert "آر مان" not in cleaned["male_names"]
        assert cleaned["male_names"].count("آرمان") == 1
        assert "امیر" in cleaned["male_names"]
        assert "ا میر" not in cleaned["male_names"]

    def test_drops_male_honorific(self) -> None:
        cleaned = clean_name_pools(self._raw())
        assert "بی بی مریم" not in cleaned["male_names"]

    def test_keeps_compounds_and_last_spaces(self) -> None:
        cleaned = clean_name_pools(self._raw())
        assert "محمد رضا" in cleaned["male_names"]
        assert "آب روشن" in cleaned["last_names"]
        assert cleaned["last_names"].count("احمدی") == 1

    def test_female_join_and_dedupe(self) -> None:
        cleaned = clean_name_pools(self._raw())
        assert "ا ناهید" not in cleaned["female_names"]
        assert cleaned["female_names"].count("آناهید") == 1

    def test_no_cross_gender_overlap_after_clean(self) -> None:
        cleaned = clean_name_pools(self._raw())
        male = set(cleaned["male_names"])
        female = set(cleaned["female_names"])
        assert male.isdisjoint(female)

    def test_output_is_sorted_unique_lists(self) -> None:
        cleaned = clean_name_pools(self._raw())
        for key, values in cleaned.items():
            assert values == sorted(set(values))

    def test_requires_expected_keys(self) -> None:
        with pytest.raises(KeyError):
            clean_name_pools({"male_names": []})


class TestPrecisionRules:
    """v1.3.0 precision repairs beyond basic OCR joins."""

    def test_join_abdol_family_simple(self) -> None:
        assert join_abdol_family("عبد الله") == "عبدالله"
        assert join_abdol_family("عبد الر ضا") == "عبدالرضا"

    def test_join_abdol_family_with_prefix(self) -> None:
        assert join_abdol_family("امیر عبد الله") == "امیر عبدالله"

    def test_join_abdol_family_noop(self) -> None:
        assert join_abdol_family("علی") == "علی"
        assert join_abdol_family("محمد رضا") == "محمد رضا"

    def test_is_truncated_name(self) -> None:
        assert is_truncated_name("اسما ال") is True
        assert is_truncated_name("عبدالله") is False
        assert is_truncated_name("") is False

    def test_is_too_short(self) -> None:
        assert is_too_short("آر") is True
        assert is_too_short("آ ر") is True
        assert is_too_short("آرش") is False
        assert is_too_short("آبث", min_letters=4) is True
        assert is_too_short("آب روشن") is False

    def test_precision_repair_pipeline(self) -> None:
        assert precision_repair("عبد الر ضا", pool_gender="male") == "عبدالرضا"
        assert precision_repair("اسما ال", pool_gender="female") is None
        assert precision_repair("آر", pool_gender="male") is None
        assert precision_repair("آرش", pool_gender="male") == "آرش"
        assert precision_repair("بی بی مریم", pool_gender="male") is None

    def test_clean_pools_applies_precision(self) -> None:
        cleaned = clean_name_pools(
            {
                "male_names": ["عبد الر ضا", "آ رمان", "آر", "علی"],
                "female_names": ["اسما ال", "فاطمه"],
                "last_names": ["محمدی", "ال"],
            }
        )
        assert cleaned["male_names"] == ["آرمان", "عبدالرضا", "علی"]
        assert cleaned["female_names"] == ["فاطمه"]
        assert cleaned["last_names"] == ["محمدی"]
