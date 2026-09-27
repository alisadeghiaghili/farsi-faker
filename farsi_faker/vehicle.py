"""Synthetic Iranian vehicle-plate generators.

Iranian car plates (پلاک خودرو), classic format: two Persian letters, a
space, and four Persian digits (e.g. ``بث ۱۲۳۴``). The province field is a
separate dimension (``car_province`` / ``vehicle_record``).

The plate *format* here is a documented model — a stable, validated shape
suitable for synthetic data — not an assertion about the exact current
official arrangement of Iranian plates.

All functions accept an optional ``random.Random`` for reproducible
fixtures.

Example:
    >>> from farsi_faker.vehicle import car_plate_number, is_valid_car_plate
    >>> plate = car_plate_number()
    >>> len(plate)
    7
    >>> is_valid_car_plate(plate)
    True
"""

from __future__ import annotations

import random
from typing import Dict, Optional

__all__ = [
    'car_plate_number',
    'is_valid_car_plate',
    'car_provinces',
    'car_province',
    'vehicle_record',
]

# Persian letters for the two-letter plate prefix — a full pool of common
# letters (all 30 standard Persian consonants plus ``و``/``ه``/``ی``).
_PLATE_LETTERS = (
    'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د',
    'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
    'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
    'ه', 'ی',
)

# Persian digits for the four-digit plate suffix.
_PERSIAN_DIGITS = (
    '۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
)

# Iran's 31 provinces — the registration-region dimension.
_PROVINCES = (
    'آذربایجان شرقی',
    'آذربایجان غربی',
    'البرز',
    'اردبیل',
    'اصفهان',
    'ایلام',
    'بوشهر',
    'تهران',
    'چهارمحال و بختیاری',
    'خراسان جنوبی',
    'خراسان رضوی',
    'خراسان شمالی',
    'خوزستان',
    'زنجان',
    'سمنان',
    'سیستان و بلوچستان',
    'فارس',
    'قزوین',
    'قم',
    'کردستان',
    'کرمان',
    'کرمانشاه',
    'کهگیلویه و بویراحمد',
    'گلستان',
    'گیلان',
    'لرستان',
    'مازندران',
    'مرکزی',
    'هرمزگان',
    'همدان',
    'یزد',
)


def _require_random(rng: Optional[random.Random]) -> random.Random:
    """Return *rng* or a fresh system Random.

    Args:
        rng (random.Random, optional): Caller-supplied RNG.

    Returns:
        random.Random: Usable RNG instance.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.
    """
    if rng is None:
        return random.Random()
    if not isinstance(rng, random.Random):
        raise TypeError(f'rng must be random.Random or None, got {type(rng)!r}')
    return rng


def is_valid_car_plate(plate: Optional[str]) -> bool:
    """Validate an Iranian vehicle plate (classic format).

    Expected shape: exactly two Persian letters, one space, exactly four
    Persian digits (U+06F0–U+06F9).

    Args:
        plate (str, optional): Candidate plate string.

    Returns:
        bool: True when the plate matches the classic Iranian plate shape.

    Example:
        >>> is_valid_car_plate('بث ۱۲۳۴')
        True
        >>> is_valid_car_plate('بث 1234')
        False
        >>> is_valid_car_plate('ب ۱۲۳۴')
        False
    """
    if not isinstance(plate, str) or len(plate) != 7:
        return False
    letters_part, sep, digits_part = plate.partition(' ')
    if sep != ' ' or len(letters_part) != 2 or len(digits_part) != 4:
        return False
    if not all(ch in _PLATE_LETTERS for ch in letters_part):
        return False
    if not all(ch in _PERSIAN_DIGITS for ch in digits_part):
        return False
    return True


def car_plate_number(rng: Optional[random.Random] = None) -> str:
    """Generate a classic Iranian vehicle plate (two letters + four digits).

    Format: ``بث ۱۲۳۴`` — two Persian letters, a space, four Persian digits.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A 7-character plate string passing :func:`is_valid_car_plate`.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> import random
        >>> plate = car_plate_number(rng=random.Random(1))
        >>> is_valid_car_plate(plate)
        True
    """
    generator = _require_random(rng)
    letters = ''.join(generator.choice(_PLATE_LETTERS) for _ in range(2))
    digits = ''.join(generator.choice(_PERSIAN_DIGITS) for _ in range(4))
    return f'{letters} {digits}'


def car_provinces() -> tuple:
    """Return the built-in pool of Iranian province names.

    Returns:
        tuple: Persian province names (immutable).

    Example:
        >>> 'تهران' in car_provinces()
        True
    """
    return _PROVINCES


def car_province(rng: Optional[random.Random] = None) -> str:
    """Pick a random Iranian province.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A province name from :func:`car_provinces`.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> car_province() in car_provinces()
        True
    """
    generator = _require_random(rng)
    return generator.choice(_PROVINCES)


def vehicle_record(rng: Optional[random.Random] = None) -> Dict[str, str]:
    """Build a synthetic vehicle record (plate + province).

    Both fields share the same RNG instance so the whole record is
    reproducible under one seed.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        Dict[str, str]: Keys ``plate`` and ``province``.

    Example:
        >>> rec = vehicle_record()
        >>> is_valid_car_plate(rec['plate'])
        True
        >>> rec['province'] in car_provinces()
        True
    """
    generator = _require_random(rng)
    return {
        'plate': car_plate_number(rng=generator),
        'province': car_province(rng=generator),
    }
