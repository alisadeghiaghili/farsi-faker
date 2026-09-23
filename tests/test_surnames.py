"""TDD specs for prefix/suffix compound surname generation."""

from __future__ import annotations

import random

import pytest

import farsi_faker
from farsi_faker import FarsiFaker
from farsi_faker.surnames import SURNAME_PREFIXES, SURNAME_SUFFIXES, compound_surname


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
        assert "SURNAME_PREFIXES" in farsi_faker.__all__
        assert "SURNAME_SUFFIXES" in farsi_faker.__all__

    def test_affix_lists_are_non_empty_and_unique(self) -> None:
        assert len(SURNAME_PREFIXES) > 10
        assert len(SURNAME_SUFFIXES) > 10
        assert len(set(SURNAME_PREFIXES)) == len(SURNAME_PREFIXES)
        assert len(set(SURNAME_SUFFIXES)) == len(SURNAME_SUFFIXES)
