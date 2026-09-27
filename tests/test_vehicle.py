"""Tests for Iranian vehicle-plate generators (M5).

The plate *format* here is a documented model (see ``farsi_faker.vehicle``),
not an assertion about the exact current official arrangement — so the tests
lock the invariants that matter for a faker: validity, reproducibility,
Persian-only glyphs, the province dimension, and the rng guard.
"""

from __future__ import annotations

import random

import pytest

from farsi_faker import FarsiFaker
from farsi_faker.vehicle import (
    car_plate_number,
    car_provinces,
    car_province,
    is_valid_car_plate,
    vehicle_record,
)

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# ---------------------------------------------------------------------------
# is_valid_car_plate
# ---------------------------------------------------------------------------


class TestIsValidCarPlate:
    """is_valid_car_plate must accept a valid model plate and reject the rest."""

    def test_accepts_valid(self) -> None:
        assert is_valid_car_plate("بث ۱۲۳۴") is True

    def test_rejects_wrong_letter_count(self) -> None:
        assert is_valid_car_plate("ب ۱۲۳۴") is False

    def test_rejects_three_letters(self) -> None:
        assert is_valid_car_plate("بثج ۱۲۳۴") is False

    def test_rejects_wrong_digit_count(self) -> None:
        assert is_valid_car_plate("بث ۱۲۳") is False

    def test_rejects_ascii_digits(self) -> None:
        assert is_valid_car_plate("بث 1234") is False

    def test_rejects_missing_space(self) -> None:
        assert is_valid_car_plate("بث۱۲۳۴") is False

    def test_rejects_digits_first(self) -> None:
        assert is_valid_car_plate("۱۲ بث") is False

    def test_rejects_non_string(self) -> None:
        assert is_valid_car_plate(None) is False  # type: ignore[arg-type]

    def test_rejects_seven_chars_no_space(self) -> None:
        # 7 chars but no separating space -> wrong structure
        assert is_valid_car_plate("بث۱۲۳۴۵") is False

    def test_rejects_letter_outside_pool(self) -> None:
        # valid shape, but the first "letter" is a digit (not in the pool)
        assert is_valid_car_plate("۰ب ۱۲۳۴") is False


# ---------------------------------------------------------------------------
# car_plate_number
# ---------------------------------------------------------------------------


class TestCarPlateNumber:
    """car_plate_number(rng) must be valid, Persian, and seedable."""

    def test_shape(self) -> None:
        # 2 letters + 1 space + 4 digits
        assert len(car_plate_number(rng=random.Random(0))) == 7

    def test_all_seeds_valid(self) -> None:
        for seed in range(100):
            assert is_valid_car_plate(car_plate_number(rng=random.Random(seed)))

    def test_persian_digits_only(self) -> None:
        digits_part = car_plate_number(rng=random.Random(1)).split(" ")[1]
        assert set(digits_part) <= set(_PERSIAN_DIGITS)

    def test_seedable(self) -> None:
        assert car_plate_number(rng=random.Random(5)) == car_plate_number(
            rng=random.Random(5)
        )

    def test_different_seeds_differ(self) -> None:
        assert car_plate_number(rng=random.Random(1)) != car_plate_number(
            rng=random.Random(2)
        )

    def test_letters_restricted_to_pool(self) -> None:
        from farsi_faker.vehicle import _PLATE_LETTERS

        for seed in range(50):
            letters = car_plate_number(rng=random.Random(seed)).split(" ")[0]
            assert set(letters) <= set(_PLATE_LETTERS)


# ---------------------------------------------------------------------------
# Province dimension
# ---------------------------------------------------------------------------


class TestCarProvince:
    """car_province(rng) / car_provinces() — the region-registered dimension."""

    def test_returns_string(self) -> None:
        assert isinstance(car_province(rng=random.Random(0)), str)

    def test_in_pool(self) -> None:
        assert car_province(rng=random.Random(1)) in car_provinces()

    def test_pool_is_tuple(self) -> None:
        assert isinstance(car_provinces(), tuple)

    def test_pool_is_substantial(self) -> None:
        # Iran has 31 provinces; assert a substantial, non-duplicated pool.
        pool = car_provinces()
        assert len(pool) >= 29
        assert len(set(pool)) == len(pool)

    def test_seedable(self) -> None:
        assert car_province(rng=random.Random(3)) == car_province(rng=random.Random(3))


# ---------------------------------------------------------------------------
# vehicle_record
# ---------------------------------------------------------------------------


class TestVehicleRecord:
    """vehicle_record(rng) -> {plate, province}, both seeded by one rng."""

    def test_keys(self) -> None:
        assert set(vehicle_record(rng=random.Random(0))) == {"plate", "province"}

    def test_plate_valid(self) -> None:
        assert is_valid_car_plate(vehicle_record(rng=random.Random(1))["plate"])

    def test_province_in_pool(self) -> None:
        assert vehicle_record(rng=random.Random(2))["province"] in car_provinces()

    def test_seedable(self) -> None:
        assert vehicle_record(rng=random.Random(7)) == vehicle_record(
            rng=random.Random(7)
        )


# ---------------------------------------------------------------------------
# rng guard
# ---------------------------------------------------------------------------


class TestRngGuard:
    @pytest.mark.parametrize(
        "generator", [car_plate_number, car_province, vehicle_record]
    )
    def test_rejects_bad_rng(self, generator) -> None:
        with pytest.raises(TypeError):
            generator(rng=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FarsiFaker convenience methods
# ---------------------------------------------------------------------------


class TestFarsiFakerVehicle:
    def test_car_plate_number(self) -> None:
        assert is_valid_car_plate(FarsiFaker(seed=42).car_plate_number())

    def test_car_province(self) -> None:
        assert FarsiFaker(seed=42).car_province() in car_provinces()

    def test_vehicle_record(self) -> None:
        rec = FarsiFaker(seed=42).vehicle_record()
        assert is_valid_car_plate(rec["plate"])
        assert rec["province"] in car_provinces()

    def test_reproducible(self) -> None:
        a = FarsiFaker(seed=99)
        b = FarsiFaker(seed=99)
        assert a.car_plate_number() == b.car_plate_number()
        assert a.vehicle_record() == b.vehicle_record()
