"""Tests for Iranian banking generators (IBAN, bank card, bank name).

Covers the Luhn mod-97 IBAN validator/generator, Luhn-10 bank card
number, and the built-in bank-name pool.
"""

from __future__ import annotations

import random

import pytest

from farsi_faker import FarsiFaker
from farsi_faker.banking import (
    bank_card_number,
    bank_name,
    iran_iban,
    iranian_banks,
    is_valid_bank_card,
    is_valid_iban,
)

# ---------------------------------------------------------------------------
# Helpers (independent of the implementation)
# ---------------------------------------------------------------------------


def _iban_check_digits(bban: str) -> str:
    """Compute the 2-digit check digits for an IR IBAN (Luhn mod 97)."""
    payload = bban + "1827" + "00"  # IR -> I(18) R(27)
    return format(98 - int(payload) % 97, "02d")


def _make_iban(bban: str) -> str:
    """Assemble a valid IR IBAN from a 24-digit bban."""
    return f"IR{_iban_check_digits(bban)}{bban}"


def _luhn_valid(number: str) -> bool:
    """Standard Luhn (mod 10) validation for a full card number."""
    total = 0
    for j, ch in enumerate(reversed(number), start=1):
        d = int(ch)
        if j % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------


class TestIsvalidIban:
    """is_valid_iban must accept structurally valid IR IBANs and reject invalid ones."""

    def test_accepts_valid(self) -> None:
        assert is_valid_iban(_make_iban("011001000000000000000100")) is True

    def test_rejects_corrupted_check_digit(self) -> None:
        good = _make_iban("011001000000000000000100")
        # Flip the last check digit
        bad = good[:-1] + ("0" if good[-1] != "0" else "1")
        assert is_valid_iban(bad) is False

    def test_rejects_non_ir_prefix(self) -> None:
        assert is_valid_iban("DE89370400440532013000") is False

    def test_rejects_wrong_length(self) -> None:
        assert is_valid_iban("IR24") is False

    def test_rejects_non_string(self) -> None:
        assert is_valid_iban(None) is False  # type: ignore[arg-type]

    def test_rejects_non_numeric_bban(self) -> None:
        assert is_valid_iban("IR00ABCDE1000000000000000000") is False

    def test_accepts_spaces_ignored(self) -> None:
        iban = _make_iban("011001000000000000000100")
        grouped = " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))
        assert is_valid_iban(grouped) is True


class TestIranIban:
    """iran_iban(rng) must produce a structurally valid, seedable IR IBAN."""

    def test_shape_28_chars(self) -> None:
        iban = iran_iban(rng=random.Random(0))
        assert len(iban) == 28

    def test_starts_with_ir(self) -> None:
        assert iran_iban(rng=random.Random(1)).startswith("IR")

    def test_check_digits_are_numeric(self) -> None:
        iban = iran_iban(rng=random.Random(2))
        assert iban[2:4].isdigit()

    def test_bban_is_24_digits(self) -> None:
        iban = iran_iban(rng=random.Random(3))
        assert iban[4:].isdigit()
        assert len(iban[4:]) == 24

    def test_validates(self) -> None:
        assert is_valid_iban(iran_iban(rng=random.Random(4))) is True

    def test_seedable(self) -> None:
        a = iran_iban(rng=random.Random(99))
        b = iran_iban(rng=random.Random(99))
        assert a == b

    def test_different_seeds_different_results(self) -> None:
        # Extremely unlikely to collide with two different seeds
        a = iran_iban(rng=random.Random(1))
        b = iran_iban(rng=random.Random(2))
        assert a != b


# ---------------------------------------------------------------------------
# Bank card number
# ---------------------------------------------------------------------------


class TestBankCardNumber:
    """bank_card_number(rng) must return a 16-digit Luhn-valid card number."""

    def test_shape_16_digits(self) -> None:
        card = bank_card_number(rng=random.Random(0))
        assert len(card) == 16
        assert card.isdigit()

    def test_luhn_valid(self) -> None:
        for seed in range(50):
            assert _luhn_valid(bank_card_number(rng=random.Random(seed)))

    def test_starts_with_iranian_prefix(self) -> None:
        # Iranian cards start with 6277-6279 (Shetab) or 6037-6219 (Shaparak)
        card = bank_card_number(rng=random.Random(5))
        assert card[:4] in (
            "6037",
            "6204",
            "6205",
            "6210",
            "6216",
            "6217",
            "6219",
            "6277",
            "6278",
            "6279",
        )

    def test_seedable(self) -> None:
        a = bank_card_number(rng=random.Random(77))
        b = bank_card_number(rng=random.Random(77))
        assert a == b


class TestIsValidBankCard:
    """is_valid_bank_card must validate Luhn-10 16-digit card numbers."""

    def test_canonical_visa(self) -> None:
        assert is_valid_bank_card("4111111111111111") is True

    def test_rejects_single_digit_change(self) -> None:
        assert is_valid_bank_card("4111111111111112") is False

    def test_rejects_wrong_length(self) -> None:
        assert is_valid_bank_card("41111111111111") is False

    def test_rejects_non_string(self) -> None:
        assert is_valid_bank_card(None) is False  # type: ignore[arg-type]

    def test_rejects_non_numeric(self) -> None:
        assert is_valid_bank_card("411111111111111X") is False

    def test_generated_cards_validate(self) -> None:
        for seed in range(30):
            assert is_valid_bank_card(bank_card_number(rng=random.Random(seed)))


class TestRngGuard:
    """All banking generators must reject non-Random rng arguments."""

    @pytest.mark.parametrize(
        "generator",
        [iran_iban, bank_card_number, bank_name],
    )
    def test_rejects_bad_rng(self, generator) -> None:
        with pytest.raises(TypeError):
            generator(rng=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bank name
# ---------------------------------------------------------------------------


class TestBankName:
    """bank_name(rng) must return a name from the built-in pool."""

    def test_returns_string(self) -> None:
        assert isinstance(bank_name(rng=random.Random(0)), str)

    def test_in_pool(self) -> None:
        assert bank_name(rng=random.Random(1)) in iranian_banks()

    def test_seedable(self) -> None:
        assert bank_name(rng=random.Random(3)) == bank_name(rng=random.Random(3))

    def test_pool_is_tuple(self) -> None:
        assert isinstance(iranian_banks(), tuple)

    def test_pool_has_multiple_entries(self) -> None:
        assert len(iranian_banks()) >= 10

    def test_pool_contains_major_banks(self) -> None:
        banks = iranian_banks()
        assert any("ملی" in b for b in banks)  # بانک ملی
        assert any("ملت" in b for b in banks)  # بانک ملت


# ---------------------------------------------------------------------------
# FarsiFaker convenience methods
# ---------------------------------------------------------------------------


class TestFarsiFakerBanking:
    """FarsiFaker.iban / .bank_card_number / .bank_name delegate correctly."""

    def test_iban(self) -> None:
        faker = FarsiFaker(seed=42)
        iban = faker.iban()
        assert is_valid_iban(iban) is True
        assert iban.startswith("IR")

    def test_bank_card_number(self) -> None:
        faker = FarsiFaker(seed=42)
        card = faker.bank_card_number()
        assert len(card) == 16
        assert _luhn_valid(card)

    def test_bank_name(self) -> None:
        faker = FarsiFaker(seed=42)
        assert faker.bank_name() in iranian_banks()

    def test_reproducible(self) -> None:
        a = FarsiFaker(seed=99)
        b = FarsiFaker(seed=99)
        assert a.iban() == b.iban()
        assert a.bank_card_number() == b.bank_card_number()
        assert a.bank_name() == b.bank_name()
