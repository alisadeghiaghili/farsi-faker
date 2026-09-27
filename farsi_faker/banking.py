"""Synthetic Iranian banking field generators.

Provides IBANs (ISO 13616, Luhn mod-97), Shetab/Shaparak bank-card numbers
(Luhn-10), and bank names — all seeded via an optional ``random.Random`` for
reproducible fixtures.

Example:
    >>> from farsi_faker.banking import iran_iban, is_valid_iban
    >>> iban = iran_iban()
    >>> len(iban)
    28
    >>> is_valid_iban(iban)
    True
"""

from __future__ import annotations

import random
import re
from typing import Optional

__all__ = [
    'iran_iban',
    'is_valid_iban',
    'bank_card_number',
    'is_valid_bank_card',
    'iranian_banks',
    'bank_name',
]

# Iranian IBAN: country code IR (I=18, R=27 -> mod-97 seed "1827"), 2 check
# digits, and a 24-digit bban. Total length 28.
_IBAN_COUNTRY = 'IR'
_IBAN_COUNTRY_SEED = '1827'
_IBAN_LENGTH = 28
_IBAN_BBAN_LENGTH = 24

# Iranian bank-card prefixes (first 4 digits). Shetab (6277-6279) and
# Shaparak (6037, 6204, 6205, 6210, 6216, 6217, 6219) card schemes.
_CARD_PREFIXES = (
    '6037',
    '6204',
    '6205',
    '6210',
    '6216',
    '6217',
    '6219',
    '6277',
    '6278',
    '6279',
)

# Built-in pool of major Iranian bank names.
_IRANIAN_BANKS = (
    'بانک ملی ایران',
    'بانک ملت',
    'بانک پارسیان',
    'بانک صادرات ایران',
    'بانک صنعت و معدن',
    'بانک کشاورزی',
    'بانک تجارت',
    'بانک سپه',
    'بانک پاسارگاد',
    'بانک اقتصاد نوین',
    'بانک سرمایه',
    'بانک شهر',
    'بانک گردشگری',
    'بانک کارآفرینی',
    'بانک دی',
    'بانک توسعه صادرات',
    'بانک قرض‌الحسنه مهر',
    'بانک قرض‌الحسنه رسالت',
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


def _iban_body_to_int(body: str) -> int:
    """Map an IBAN body (``A-Z0-9``) to its base-36 integer value.

    Args:
        body (str): Uppercase IBAN fragment.

    Returns:
        int: Numeric value after the ISO 7064 character mapping.
    """
    return int(''.join(str(int(char, 36)) for char in body.upper()))


def is_valid_iban(iban: Optional[str]) -> bool:
    """Validate an Iranian (IR) IBAN using the ISO 13616 Luhn mod-97 scheme.

    Args:
        iban (str, optional): Candidate IBAN. Internal spaces are ignored.

    Returns:
        bool: True when well-formed (IR prefix, 28 characters, numeric bban)
        and the Luhn mod-97 check passes.

    Example:
        >>> is_valid_iban('IR49000000000000000000000000')
        True
        >>> is_valid_iban('IR00000000000000000000000000')
        False
    """
    if not isinstance(iban, str):
        return False
    cleaned = iban.replace(' ', '').upper()
    if not cleaned.startswith(_IBAN_COUNTRY) or len(cleaned) != _IBAN_LENGTH:
        return False
    bban = cleaned[4:]
    if not bban.isdigit():
        return False
    # Move the country+check field to the end, then test divisibility by 97.
    rearranged = bban + cleaned[:4]
    return _iban_body_to_int(rearranged) % 97 == 1


def _iban_check_digits(bban: str) -> str:
    """Compute the 2-digit check field for an IR IBAN from a bban.

    Args:
        bban (str): 24-digit body.

    Returns:
        str: Two-digit check field.
    """
    check = 98 - (int(bban + _IBAN_COUNTRY_SEED + '00') % 97)
    return format(check, '02d')


def iran_iban(rng: Optional[random.Random] = None) -> str:
    """Generate a structurally valid Iranian (IR) IBAN.

    The bban is a random 24-digit string; the check digits are computed so
    the result passes :func:`is_valid_iban`.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: 28-character IBAN starting with ``IR``.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> import random
        >>> len(iran_iban(rng=random.Random(1)))
        28
    """
    generator = _require_random(rng)
    bban = ''.join(str(generator.randint(0, 9)) for _ in range(_IBAN_BBAN_LENGTH))
    return f'{_IBAN_COUNTRY}{_iban_check_digits(bban)}{bban}'


def _luhn_check_digit(number: str) -> int:
    """Compute the Luhn (mod-10) check digit to append to *number*.

    Args:
        number (str): Numeric body (the check digit is not included).

    Returns:
        int: The single check digit in ``[0, 9]``.
    """
    total = 0
    for index, char in enumerate(reversed(number), start=1):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return (10 - (total % 10)) % 10


def is_valid_bank_card(number: Optional[str]) -> bool:
    """Validate a 16-digit bank-card number with the Luhn (mod-10) checksum.

    Args:
        number (str, optional): Candidate 16-digit card number.

    Returns:
        bool: True when the number is 16 digits and passes Luhn.

    Example:
        >>> is_valid_bank_card('4111111111111111')
        True
        >>> is_valid_bank_card('4111111111111112')
        False
    """
    if not isinstance(number, str) or not re.fullmatch(r'\d{16}', number):
        return False
    total = 0
    for index, char in enumerate(reversed(number), start=1):
        digit = int(char)
        if index % 2 == 0:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def bank_card_number(rng: Optional[random.Random] = None) -> str:
    """Generate a 16-digit Luhn-valid Iranian bank-card number.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A 16-digit card number starting with an Iranian prefix and
        passing :func:`is_valid_bank_card`.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> import random
        >>> card = bank_card_number(rng=random.Random(1))
        >>> len(card)
        16
    """
    generator = _require_random(rng)
    prefix = generator.choice(_CARD_PREFIXES)
    body = prefix + ''.join(str(generator.randint(0, 9)) for _ in range(11))
    return body + str(_luhn_check_digit(body))


def iranian_banks() -> tuple:
    """Return the built-in pool of major Iranian bank names.

    Returns:
        tuple: Persian bank names (immutable).

    Example:
        >>> 'بانک ملت' in iranian_banks()
        True
    """
    return _IRANIAN_BANKS


def bank_name(rng: Optional[random.Random] = None) -> str:
    """Pick a random Iranian bank name.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A bank name from :func:`iranian_banks`.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> bank_name() in iranian_banks()
        True
    """
    generator = _require_random(rng)
    return generator.choice(_IRANIAN_BANKS)
