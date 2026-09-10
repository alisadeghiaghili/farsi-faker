"""TDD specs for Iranian synthetic profile field generators."""

from __future__ import annotations

import re

import pytest

from farsi_faker.profile import (
    email_address,
    is_valid_mobile,
    is_valid_national_id,
    mobile_number,
    national_id,
    postal_code,
    profile_record,
)


class TestNationalId:
    """کد ملی — 10-digit with official checksum."""

    def test_length_and_digits(self) -> None:
        code = national_id()
        assert re.fullmatch(r"\d{10}", code)

    def test_checksum_valid(self) -> None:
        for _ in range(50):
            code = national_id()
            assert is_valid_national_id(code) is True

    def test_known_valid_codes(self) -> None:
        assert is_valid_national_id("1111111111") is True
        assert is_valid_national_id("1043321810") is True

    def test_invalid_codes_rejected(self) -> None:
        assert is_valid_national_id("0013540398") is False
        assert is_valid_national_id("123") is False
        assert is_valid_national_id("abcdefghij") is False
        assert is_valid_national_id("") is False

    def test_reproducible_with_seed(self) -> None:
        import random

        rng_a = random.Random(42)
        rng_b = random.Random(42)
        assert national_id(rng=rng_a) == national_id(rng=rng_b)

    def test_rejects_bad_rng(self) -> None:
        with pytest.raises(TypeError):
            national_id(rng="nope")  # type: ignore[arg-type]


class TestMobileNumber:
    """Iranian mobile numbers: 09xxxxxxxxx."""

    def test_format(self) -> None:
        number = mobile_number()
        assert re.fullmatch(r"09\d{9}", number)
        assert len(number) == 11

    def test_known_operator_prefixes(self) -> None:
        for _ in range(30):
            assert is_valid_mobile(mobile_number()) is True

    def test_invalid_mobile(self) -> None:
        assert is_valid_mobile("08123456789") is False
        assert is_valid_mobile("091234567") is False
        assert is_valid_mobile("091234567890") is False
        assert is_valid_mobile("") is False

    def test_reproducible_with_seed(self) -> None:
        import random

        assert mobile_number(rng=random.Random(7)) == mobile_number(rng=random.Random(7))


class TestEmail:
    """Email derived from person name parts."""

    def test_basic_shape(self) -> None:
        email = email_address(first_name="علی", last_name="احمدی")
        assert "@" in email
        local, domain = email.split("@", 1)
        assert local
        assert "." in domain
        assert re.fullmatch(r"[a-z0-9._-]+", local)

    def test_uses_romanized_slug(self) -> None:
        email = email_address(first_name="Ali", last_name="Ahmadi")
        assert email.startswith("ali") or "ali" in email.split("@")[0]
        assert "ahmadi" in email

    def test_without_names_still_valid(self) -> None:
        email = email_address()
        assert re.fullmatch(r"[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}", email)

    def test_reproducible(self) -> None:
        import random

        a = email_address(rng=random.Random(1))
        b = email_address(rng=random.Random(1))
        assert a == b


class TestPostalCode:
    """10-digit Iranian postal code."""

    def test_format(self) -> None:
        code = postal_code()
        assert re.fullmatch(r"\d{10}", code)


class TestProfileRecord:
    """Composite person fixture."""

    def test_keys_and_consistency(self) -> None:
        person = profile_record(seed=42)
        assert {
            "name",
            "first_name",
            "last_name",
            "gender",
            "national_id",
            "mobile",
            "email",
            "postal_code",
        }.issubset(person.keys())
        assert person["name"] == f"{person['first_name']} {person['last_name']}"
        assert is_valid_national_id(person["national_id"])
        assert is_valid_mobile(person["mobile"])
        assert "@" in person["email"]

    def test_gender_filter(self) -> None:
        person = profile_record(gender="female", seed=1)
        assert person["gender"] == "female"

    def test_reproducible(self) -> None:
        assert profile_record(seed=99) == profile_record(seed=99)
