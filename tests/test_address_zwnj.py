"""TDD specs for ZWNJ normalization and address/city generators."""

from __future__ import annotations

import random
import re

from farsi_faker.cleaning import normalize_zwnj, apply_zwnj_policy
from farsi_faker.profile import (
    address_record,
    city_name,
    iranian_cities,
    street_name,
)


class TestNormalizeZwnj:
    """ZWNJ / space hygiene."""

    def test_collapses_zwnj_beside_space(self) -> None:
        # Space around ZWNJ should not leave double separators.
        assert normalize_zwnj("می‌ رود") == "می‌رود"
        assert normalize_zwnj("می ‌رود") == "می‌رود"

    def test_strips_leading_trailing_zwnj(self) -> None:
        assert normalize_zwnj("‌علی‌") == "علی"

    def test_preserves_internal_zwnj(self) -> None:
        assert normalize_zwnj("می‌رود") == "می‌رود"

    def test_empty(self) -> None:
        assert normalize_zwnj("") == ""
        assert normalize_zwnj(None) == ""  # type: ignore[arg-type]


class TestApplyZwnjPolicy:
    """Convert selected space-separated compounds to ZWNJ form."""

    def test_known_compounds_get_zwnj(self) -> None:
        assert apply_zwnj_policy("می خواهد") == "می‌خواهد"
        assert apply_zwnj_policy("خانه دار") == "خانه‌دار"

    def test_unknown_compounds_untouched(self) -> None:
        assert apply_zwnj_policy("محمد رضا") == "محمد رضا"
        assert apply_zwnj_policy("علی") == "علی"

    def test_empty(self) -> None:
        assert apply_zwnj_policy("") == ""


class TestCities:
    """Iranian city pool."""

    def test_contains_major_cities(self) -> None:
        cities = iranian_cities()
        assert "تهران" in cities
        assert "مشهد" in cities
        assert "اصفهان" in cities
        assert "شیراز" in cities
        assert "تبریز" in cities

    def test_size_reasonable(self) -> None:
        assert len(iranian_cities()) >= 20

    def test_city_name_uses_pool(self) -> None:
        assert city_name() in iranian_cities()

    def test_city_name_reproducible(self) -> None:
        a = city_name(rng=random.Random(5))
        b = city_name(rng=random.Random(5))
        assert a == b


class TestAddress:
    """Street and full address records."""

    def test_street_format(self) -> None:
        street = street_name()
        assert street
        assert isinstance(street, str)

    def test_address_record_keys(self) -> None:
        addr = address_record(rng=random.Random(1))
        assert {"city", "street", "alley", "plaque", "postal_code"}.issubset(addr.keys())
        assert addr["city"] in iranian_cities()
        assert re.fullmatch(r"\d{10}", addr["postal_code"])
        assert addr["plaque"].isdigit()

    def test_address_reproducible(self) -> None:
        a = address_record(rng=random.Random(9))
        b = address_record(rng=random.Random(9))
        assert a == b
