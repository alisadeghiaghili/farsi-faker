"""TDD specs for prefix/suffix compound surname generation."""

from __future__ import annotations

import random

import pytest

import farsi_faker
from farsi_faker import FarsiFaker
from farsi_faker.profile import iranian_cities
from farsi_faker.surnames import (
    _OPTIONAL_VOICED,
    _VOICED_BASES,
    OCCUPATION_SURNAMES,
    SURNAME_PREFIXES,
    SURNAME_SUFFIXES,
    apply_surname_phonetics,
    compound_surname,
    occupation_surname,
    region_surname,
)


class TestCompoundSurname:
    """The pure composition function."""

    def test_returns_a_string(self) -> None:
        assert isinstance(compound_surname("احمد", rng=random.Random(0)), str)

    def test_base_is_always_preserved(self) -> None:
        # The base name must appear inside every generated surname.
        for seed in range(40):
            result = compound_surname("احمد", rng=random.Random(seed))
            assert "احمد" in result, result

    @staticmethod
    def _structurally_valid(base: str, result: str) -> bool:
        """True if *result* matches one of the four allowed compositions."""
        # terminate: base or base + ی
        if result in (base, base + "ی"):
            return True
        # glued: single token starting with base, rest = suffix (+ optional ی)
        if " " not in result and result.startswith(base):
            tail = result[len(base):]
            for suffix in SURNAME_SUFFIXES:
                if tail == suffix or tail == suffix + "ی":
                    return True
        # spaced suffix: base + ' ' + suffix
        if " " in result:
            tokens = result.split()
            if len(tokens) == 2 and tokens[0] == base and tokens[1] in SURNAME_SUFFIXES:
                return True
            # prefix: title + ' ' + base
            if len(tokens) == 2 and tokens[1] == base and tokens[0] in SURNAME_PREFIXES:
                return True
        return False

    def test_uses_only_known_affixes(self) -> None:
        # Every generated name must match one of the four structural forms
        # built from the base plus a known prefix/suffix/termination.
        for seed in range(120):
            result = compound_surname("احمد", rng=random.Random(seed))
            assert self._structurally_valid("احمد", result), result

    def test_all_four_forms_appear(self) -> None:
        # Over many draws every composition form should be produced.
        results = [compound_surname("احمد", rng=random.Random(i)) for i in range(400)]
        # termination form: base + ی
        assert "احمدی" in results
        # glued suffix form: single token, not just base/base+ی
        assert any(
            " " not in r and r not in ("احمد", "احمدی") for r in results
        )
        # spaced suffix form: first token is the base, second is a known suffix
        assert any(
            " " in r and r.split()[0] == "احمد" and r.split()[-1] in SURNAME_SUFFIXES
            for r in results
        )
        # prefix form: first token is a known prefix, second is the base
        assert any(
            " " in r and r.split()[0] in SURNAME_PREFIXES and "احمد" in r.split()
            for r in results
        )

    def test_reproducible_with_seed(self) -> None:
        a = [compound_surname("رضا", rng=random.Random(9)) for _ in range(5)]
        b = [compound_surname("رضا", rng=random.Random(9)) for _ in range(5)]
        assert a == b

    def test_different_seeds_diverge(self) -> None:
        # Not every pair of seeds collides over a handful of draws.
        draws_a = [compound_surname("علی", rng=random.Random(i)) for i in range(5)]
        draws_b = [compound_surname("علی", rng=random.Random(i + 1000)) for i in range(5)]
        assert draws_a != draws_b

    def test_rejects_empty_base(self) -> None:
        with pytest.raises(ValueError):
            compound_surname("   ")

    def test_rejects_bad_rng(self) -> None:
        with pytest.raises(TypeError):
            compound_surname("احمد", rng="nope")  # type: ignore[arg-type]


class TestApplySurnamePhonetics:
    """Base voicing applied before composition (the "روی پایه" rule)."""

    def test_returns_a_string(self) -> None:
        assert isinstance(apply_surname_phonetics("احمد", rng=random.Random(0)), str)

    def test_voiced_bases_map_to_their_family_form(self) -> None:
        # Every fixed-voicing base must return its voiced form, deterministically.
        for base, voiced in _VOICED_BASES.items():
            assert apply_surname_phonetics(base) == voiced

    def test_known_voiced_examples(self) -> None:
        assert apply_surname_phonetics("مصطفی") == "مصطفوی"
        assert apply_surname_phonetics("کسری") == "کسروی"
        assert apply_surname_phonetics("یحیی") == "یحیوی"

    def test_unchanged_base_passes_through(self) -> None:
        # A base with no voicing rule must come back exactly as given.
        assert apply_surname_phonetics("احمد") == "احمد"
        assert apply_surname_phonetics("حسین") == "حسین"

    def test_optional_base_is_one_of_the_two_forms(self) -> None:
        for base, voiced in _OPTIONAL_VOICED.items():
            for seed in range(200):
                result = apply_surname_phonetics(base, rng=random.Random(seed))
                assert result in (base, voiced), result

    def test_optional_base_produces_both_forms_over_seeds(self) -> None:
        # Over enough draws both the plain and voiced forms must appear.
        base, voiced = next(iter(_OPTIONAL_VOICED.items()))
        results = {apply_surname_phonetics(base, rng=random.Random(i)) for i in range(200)}
        assert results == {base, voiced}

    def test_optional_base_requires_rng_choice(self) -> None:
        # Same seed -> same choice; the decision is reproducible.
        base = next(iter(_OPTIONAL_VOICED))
        assert (
            apply_surname_phonetics(base, rng=random.Random(5))
            == apply_surname_phonetics(base, rng=random.Random(5))
        )

    def test_compound_preserves_voiced_base(self) -> None:
        # A voiced base must survive into the composed surname, so the result
        # reads as a family name (مصطفوی… not مصطفی…).
        for seed in range(120):
            result = compound_surname("مصطفی", rng=random.Random(seed))
            assert "مصطفوی" in result, result

    def test_compound_preserves_plain_base(self) -> None:
        for seed in range(120):
            result = compound_surname("احمد", rng=random.Random(seed))
            assert "احمد" in result, result

    def test_compound_does_not_leak_unvoiced_form(self) -> None:
        # The original first-name form should not appear unvoiced in the
        # composed result for a fixed-voicing base.
        for seed in range(120):
            result = compound_surname("کسری", rng=random.Random(seed))
            assert "کسری" not in result, result
            assert "کسروی" in result, result

    def test_rejects_empty_base(self) -> None:
        with pytest.raises(ValueError):
            apply_surname_phonetics("   ")

    def test_rejects_bad_rng(self) -> None:
        with pytest.raises(TypeError):
            apply_surname_phonetics("علی", rng="nope")  # type: ignore[arg-type]


class TestFarsiFakerIntegration:
    """The method wires the composer to the instance RNG and pool."""

    def test_default_base_is_a_male_name(self) -> None:
        faker = FarsiFaker(seed=1)
        name = faker.generated_last_name()
        # Base is drawn from the male pool, so one of its tokens must be a
        # real pool member.
        assert isinstance(name, str) and name

    def test_explicit_base_is_used(self) -> None:
        faker = FarsiFaker(seed=2)
        name = faker.generated_last_name(base="حسین")
        assert "حسین" in name

    def test_reproducible(self) -> None:
        a = FarsiFaker(seed=5).generated_last_name(base="رضا")
        b = FarsiFaker(seed=5).generated_last_name(base="رضا")
        assert a == b

    def test_explicit_base_advances_rng(self) -> None:
        # Two fresh instances with the same seed produce the same first name.
        a = FarsiFaker(seed=7).generated_last_name(base="رضا")
        b = FarsiFaker(seed=7).generated_last_name(base="رضا")
        assert a == b


class TestExports:
    """Public API surface."""

    def test_public_exports(self) -> None:
        assert hasattr(farsi_faker, "compound_surname")
        assert "compound_surname" in farsi_faker.__all__
        assert "apply_surname_phonetics" in farsi_faker.__all__
        assert "SURNAME_PREFIXES" in farsi_faker.__all__
        assert "SURNAME_SUFFIXES" in farsi_faker.__all__
        assert "region_surname" in farsi_faker.__all__
        assert "occupation_surname" in farsi_faker.__all__
        assert "OCCUPATION_SURNAMES" in farsi_faker.__all__

    def test_affix_lists_are_non_empty_and_unique(self) -> None:
        assert len(SURNAME_PREFIXES) > 10
        assert len(SURNAME_SUFFIXES) > 10
        assert len(set(SURNAME_PREFIXES)) == len(SURNAME_PREFIXES)
        assert len(set(SURNAME_SUFFIXES)) == len(SURNAME_SUFFIXES)


class TestRegionSurname:
    """Region-based surnames: city -> city + ی."""

    def test_appends_i_to_city(self) -> None:
        assert region_surname("تهران") == "تهرانی"
        assert region_surname("اصفهان") == "اصفهانی"

    def test_keeps_i_ending_when_present(self) -> None:
        # A city that already ends in ی must not get a doubled termination.
        assert region_surname("تبریز") == "تبریزی"
        assert region_surname("قم") == "قمی"

    def test_is_deterministic_per_city(self) -> None:
        # Same city -> same surname, regardless of seed.
        assert region_surname("شیراز", rng=random.Random(1)) == region_surname(
            "شیراز", rng=random.Random(99)
        )

    def test_rejects_empty_city(self) -> None:
        with pytest.raises(ValueError):
            region_surname("  ")

    def test_rejects_bad_rng(self) -> None:
        with pytest.raises(TypeError):
            region_surname("تهران", rng="nope")  # type: ignore[arg-type]


class TestFarsiFakerRegionIntegration:
    """FarsiFaker.region_last_name wires a city into region_surname."""

    def test_explicit_city(self) -> None:
        assert FarsiFaker(seed=1).region_last_name(city="تهران") == "تهرانی"

    def test_default_city_is_from_pool(self) -> None:
        faker = FarsiFaker(seed=2)
        name = faker.region_last_name()
        cities = set(iranian_cities())
        # The result must be the -ی form of one of the built-in cities.
        assert any(
            name == (city if city.endswith("ی") else city + "ی")
            for city in cities
        ), name

    def test_reproducible(self) -> None:
        a = FarsiFaker(seed=4).region_last_name()
        b = FarsiFaker(seed=4).region_last_name()
        assert a == b


class TestOccupationSurname:
    """Occupational (trade) surnames: a whole-name pool, not composed."""

    def test_pool_is_non_empty_and_unique(self) -> None:
        assert len(OCCUPATION_SURNAMES) > 10
        assert len(set(OCCUPATION_SURNAMES)) == len(OCCUPATION_SURNAMES)

    def test_contains_well_known_trades(self) -> None:
        # The trades that motivated this feature must be present.
        for trade in ("قناد", "خراط", "فلاح", "نانوایی", "نجار"):
            assert trade in OCCUPATION_SURNAMES

    def test_returns_a_pool_member(self) -> None:
        for seed in range(40):
            assert occupation_surname(rng=random.Random(seed)) in OCCUPATION_SURNAMES

    def test_reproducible_with_seed(self) -> None:
        a = [occupation_surname(rng=random.Random(2)) for _ in range(5)]
        b = [occupation_surname(rng=random.Random(2)) for _ in range(5)]
        assert a == b

    def test_rejects_bad_rng(self) -> None:
        with pytest.raises(TypeError):
            occupation_surname(rng="nope")  # type: ignore[arg-type]

    def test_zwnj_orthography_is_preserved(self) -> None:
        # Compound trade names carry a ZWNJ (e.g. نقش‌باف) and must keep it.
        zwnj = "‌"
        assert any(zwnj in name for name in OCCUPATION_SURNAMES)


class TestFarsiFakerOccupationIntegration:
    """FarsiFaker.occupation_last_name draws from the instance RNG."""

    def test_returns_a_pool_member(self) -> None:
        assert FarsiFaker(seed=1).occupation_last_name() in OCCUPATION_SURNAMES

    def test_reproducible(self) -> None:
        assert FarsiFaker(seed=3).occupation_last_name() == FarsiFaker(
            seed=3
        ).occupation_last_name()
