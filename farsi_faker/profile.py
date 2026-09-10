"""Synthetic Iranian profile field generators.

Provides checksum-valid national IDs (کد ملی), Iranian mobile numbers,
emails derived from Persian names, and postal codes — all seeded via an
optional ``random.Random`` for reproducible fixtures.

Example:
    >>> from farsi_faker.profile import national_id, mobile_number, is_valid_national_id
    >>> code = national_id()
    >>> len(code)
    10
    >>> is_valid_national_id(code)
    True
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, Optional

__all__ = [
    'national_id',
    'is_valid_national_id',
    'mobile_number',
    'is_valid_mobile',
    'email_address',
    'postal_code',
    'profile_record',
]

# Common Iranian mobile operator prefixes (first three digits after 0).
_MOBILE_PREFIXES = (
    '090',
    '091',
    '092',
    '093',
    '099',
)

_EMAIL_DOMAINS = (
    'gmail.com',
    'yahoo.com',
    'outlook.com',
    'mail.ir',
    'chmail.ir',
)

# Minimal romanization map for common Persian letters used in emails.
_FA_TO_EN = str.maketrans(
    {
        'ا': 'a',
        'ب': 'b',
        'پ': 'p',
        'ت': 't',
        'ث': 's',
        'ج': 'j',
        'چ': 'ch',
        'ح': 'h',
        'خ': 'kh',
        'د': 'd',
        'ذ': 'z',
        'ر': 'r',
        'ز': 'z',
        'ژ': 'zh',
        'س': 's',
        'ش': 'sh',
        'ص': 's',
        'ض': 'z',
        'ط': 't',
        'ظ': 'z',
        'ع': 'a',
        'غ': 'gh',
        'ف': 'f',
        'ق': 'gh',
        'ک': 'k',
        'گ': 'g',
        'ل': 'l',
        'م': 'm',
        'ن': 'n',
        'و': 'v',
        'ه': 'h',
        'ی': 'y',
        'آ': 'a',
        'ء': '',
        ' ': '.',
        '‌': '',  # ZWNJ
    }
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


def _national_id_checksum(digits: str) -> int:
    """Compute the official Iranian national-ID check digit.

    Args:
        digits (str): The first 9 digits as a string.

    Returns:
        int: Check digit in ``[0, 10]``.
    """
    total = sum(int(digit) * (10 - index) for index, digit in enumerate(digits))
    remainder = total % 11
    return remainder if remainder < 2 else 11 - remainder


def is_valid_national_id(code: str) -> bool:
    """Validate an Iranian national ID (کد ملی) including checksum.

    Args:
        code (str): Candidate code. Digits only, length 10.

    Returns:
        bool: True when the code is well-formed and the checksum matches.

    Example:
        >>> is_valid_national_id('0013540399')
        True
        >>> is_valid_national_id('0013540398')
        False
    """
    if not isinstance(code, str) or not re.fullmatch(r'\d{10}', code):
        return False
    return _national_id_checksum(code[:9]) == int(code[9])


def national_id(rng: Optional[random.Random] = None) -> str:
    """Generate a checksum-valid 10-digit Iranian national ID.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A 10-digit national ID that passes :func:`is_valid_national_id`.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> import random
        >>> code = national_id(rng=random.Random(0))
        >>> len(code)
        10
        >>> is_valid_national_id(code)
        True
    """
    generator = _require_random(rng)
    # Avoid leading-all-zero edge cases that look fake; allow a leading zero
    # when the remaining digits are non-trivial.
    body = [generator.randint(0, 9) for _ in range(9)]
    if all(digit == 0 for digit in body):
        body[-1] = generator.randint(1, 9)
    digits = ''.join(str(digit) for digit in body)
    return f'{digits}{_national_id_checksum(digits)}'


def is_valid_mobile(number: str) -> bool:
    """Validate Iranian mobile number shape and prefix.

    Args:
        number (str): Candidate number.

    Returns:
        bool: True for ``09xxxxxxxxx`` with a known operator prefix.

    Example:
        >>> is_valid_mobile('09123456789')
        True
        >>> is_valid_mobile('08123456789')
        False
    """
    if not isinstance(number, str) or not re.fullmatch(r'09\d{9}', number):
        return False
    return number[:3] in _MOBILE_PREFIXES


def mobile_number(rng: Optional[random.Random] = None) -> str:
    """Generate a syntactically valid Iranian mobile number.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: 11-digit number starting with a known ``09x`` operator prefix.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> import random
        >>> number = mobile_number(rng=random.Random(1))
        >>> len(number)
        11
        >>> is_valid_mobile(number)
        True
    """
    generator = _require_random(rng)
    prefix = generator.choice(_MOBILE_PREFIXES)
    suffix = ''.join(str(generator.randint(0, 9)) for _ in range(8))
    return f'{prefix}{suffix}'


def _slugify(text: str) -> str:
    """Romanize and slugify a name fragment for email local-parts.

    Args:
        text (str): Persian or ASCII name fragment.

    Returns:
        str: Lowercase ``[a-z0-9._-]`` slug; may be empty.
    """
    translated = text.translate(_FA_TO_EN).lower()
    slug = re.sub(r'[^a-z0-9._-]+', '', translated)
    slug = re.sub(r'[._-]{2,}', '.', slug).strip('.-_')
    return slug


def email_address(
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Build a synthetic email, optionally from person name parts.

    Args:
        first_name (str, optional): Used to build the local-part.
        last_name (str, optional): Used to build the local-part.
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: Email matching ``local@domain.tld``.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> email_address(first_name='Ali', last_name='Ahmadi')
        'ali.ahmadi@gmail.com'
    """
    generator = _require_random(rng)
    parts = []
    if first_name:
        slug = _slugify(first_name)
        if slug:
            parts.append(slug)
    if last_name:
        slug = _slugify(last_name)
        if slug:
            parts.append(slug)

    if not parts:
        handle = ''.join(generator.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(8))
        parts.append(handle)

    local = '.'.join(parts)
    domain = generator.choice(_EMAIL_DOMAINS)
    return f'{local}@{domain}'


def postal_code(rng: Optional[random.Random] = None) -> str:
    """Generate a 10-digit Iranian postal code.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A 10-digit numeric postal code.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> code = postal_code()
        >>> len(code)
        10
    """
    generator = _require_random(rng)
    return ''.join(str(generator.randint(0, 9)) for _ in range(10))


def profile_record(
    gender: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a full synthetic person record (name + contact fields).

    Args:
        gender (str, optional): Passed through to :class:`~farsi_faker.FarsiFaker`.
        seed (int, optional): Seed for a private ``random.Random`` shared by
            all fields so the whole record is reproducible.

    Returns:
        Dict[str, Any]: Keys ``name``, ``first_name``, ``last_name``,
        ``gender``, ``national_id``, ``mobile``, ``email``, ``postal_code``.

    Example:
        >>> person = profile_record(gender='male', seed=42)
        >>> person['gender']
        'male'
        >>> '@' in person['email']
        True
    """
    from .faker import FarsiFaker

    generator = random.Random(seed)
    faker = FarsiFaker(seed=generator.randint(0, 2**32 - 1))
    person = faker.full_name(gender)

    return {
        **person,
        'national_id': national_id(rng=generator),
        'mobile': mobile_number(rng=generator),
        'email': email_address(
            first_name=person['first_name'],
            last_name=person['last_name'],
            rng=generator,
        ),
        'postal_code': postal_code(rng=generator),
    }
