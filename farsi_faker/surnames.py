"""Generated Persian family names via prefix/suffix composition.

This implements the well-known ``persian-names`` approach — building a family
name from a base name plus a common Persian title prefix and/or surname
suffix — but done cleanly: it is seedable, reuses the package's canonical
whitespace normalizer, and avoids the fragile "slice the name list by file
line number" heuristic the original relies on (which silently breaks whenever
the data file is reordered).

Examples of the composition forms produced:

* terminate:  ``احمد`` -> ``احمدی``
* glued:      ``احمد`` -> ``احمدپور`` / ``احمدپوری``
* spaced:     ``احمد`` -> ``احمد پور``
* prefix:     ``احمد`` -> ``میر احمد``

All of these are valid, realistic Persian family names.
"""

from __future__ import annotations

import random
from typing import Optional

from .cleaning import normalize_name

__all__ = [
    'SURNAME_PREFIXES',
    'SURNAME_SUFFIXES',
    'OCCUPATION_SURNAMES',
    'compound_surname',
    'region_surname',
    'occupation_surname',
]

# Common Persian title/name prefixes used at the head of a family name.
# حاج / حاجی are kept: they are legitimate (unisex) surname prefixes too.
SURNAME_PREFIXES = (
    'میر',
    'پیر',
    'یار',
    'سید',
    'امیر',
    'عزیز',
    'صیاد',
    'زاهد',
    'شاه',
    'نیک',
    'حاج',
    'حاجی',
    'صوفی',
    'افضل',
    'فاضل',
    'شیخ',
    'میرزا',
    'استاد',
    'خواجه',
    'ملک',
    'خان',
    'بیگ',
    'عرب',
    'منش',
)

# Common Persian surname suffixes attached to a base name.
SURNAME_SUFFIXES = (
    'پور',
    'زاده',
    'فر',
    'فرد',
    'کیا',
    'راد',
    'زند',
    'خواه',
    'نیا',
    'مهر',
    'آذر',
    'صدر',
    'کهن',
    'نژاد',
    'بیات',
    'یکتا',
    'ثابت',
    'ازاد',
    'زارع',
    'مقدم',
    'روشان',
    'تبار',
    'راشد',
    'دانا',
    'زادگان',
    'منش',
    'یار',
)

# Occupational / trade-based family names — Persian surnames that derive from
# a person's trade or craft (e.g. قناد "confectioner", خراط "woodturner",
# فلاح "farmer"). These are used as whole surnames, not composed.
# Spelling uses the canonical ZWNJ for compound forms (نقش‌باف، شیشه‌گر).
OCCUPATION_SURNAMES = (
    'فلاح',
    'کشاورز',
    'باغبان',
    'قناد',
    'نانوایی',
    'خراط',
    'نجار',
    'مسگر',
    'زرگر',
    'طلاگر',
    'کفاش',
    'بافند',
    'نقش‌باف',
    'شیشه‌گر',
    'آهنگر',
    'معلم',
    'پزشک',
    'داروساز',
    'حسابدار',
    'مهندس',
    'دباغ',
    'بازرگان',
    'قصاب',
    'کتابدار',
    'چوپان',
    'زنبوردار',
    'ساعت‌ساز',
    'قفل‌ساز',
    'آینه‌ساز',
    'بستنی‌ساز',
)

# The common Persian surname termination (the "-i"/"ی" ending, e.g. احمدی).
_TERMINATION = 'ی'

# (pattern, weight): relative frequency of each composition form.
_PATTERNS = (
    ('terminate', 3),  # base + ی            (احمد -> احمدی)
    ('glued', 3),      # base + suffix [+ ی] (احمد -> احمدپور / احمدپوری)
    ('spaced', 2),     # base + ' ' + suffix (احمد -> احمد پور)
    ('prefix', 2),     # title + ' ' + base  (میر احمد)
)
_TOTAL_WEIGHT = sum(weight for _, weight in _PATTERNS)


def _require_random(rng: Optional[random.Random]) -> random.Random:
    """Return *rng* or a fresh system Random; validate the type.

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


def _weighted_pattern(rng: random.Random) -> str:
    """Pick a composition pattern by weight.

    Args:
        rng (random.Random): RNG to draw from.

    Returns:
        str: One of the pattern names in :data:`_PATTERNS`.
    """
    remaining = rng.randint(1, _TOTAL_WEIGHT)
    for pattern, weight in _PATTERNS:
        remaining -= weight
        if remaining <= 0:
            return pattern
    return _PATTERNS[0][0]


def compound_surname(base: str, *, rng: Optional[random.Random] = None) -> str:
    """Compose a Persian family name from a base name and a prefix/suffix.

    The base is always preserved inside the result. One of four composition
    forms is chosen (weighted): a title prefix before the base, a space
    separated suffix after it, a glued suffix (optionally terminated in
    ``ی``), or the ``ی`` termination alone.

    Args:
        base (str): A base name token (typically a first name). Must be
            non-empty.
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: A composed family name with whitespace normalised.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.
        ValueError: If *base* is empty after stripping.

    Example:
        >>> import random
        >>> name = compound_surname('احمد', rng=random.Random(7))
        >>> isinstance(name, str)
        True
        >>> 'احمد' in name
        True
    """
    generator = _require_random(rng)
    base = base.strip()
    if not base:
        raise ValueError('base must be a non-empty name')

    suffix = generator.choice(SURNAME_SUFFIXES)
    pattern = _weighted_pattern(generator)

    if pattern == 'terminate':
        result = base if base.endswith(_TERMINATION) else base + _TERMINATION
    elif pattern == 'glued':
        result = base + suffix
        if generator.random() < 0.5:
            result += _TERMINATION
    elif pattern == 'spaced':
        result = f'{base} {suffix}'
    else:  # prefix
        result = f'{generator.choice(SURNAME_PREFIXES)} {base}'

    return normalize_name(result)


def region_surname(city: str, *, rng: Optional[random.Random] = None) -> str:
    """Turn an Iranian city name into a region-based family name.

    This produces the common "X-i" regional surname from a place name, e.g.
    ``تهران`` -> ``تهرانی``, ``اصفهان`` -> ``اصفهانی``, ``تبریز`` ->
    ``تبریزی``. Cities that already end in the termination are returned
    unchanged.

    The rule is deterministic, so the same city always yields the same
    surname; ``rng`` is accepted (and validated) for signature consistency
    with the other generators so the interface can later draw a variant
    (e.g. the rare ``-آبادی`` form) without an API change.

    Args:
        city (str): A Persian city / place name. Must be non-empty.
        rng (random.Random, optional): Seeded RNG (validated; not yet used).

    Returns:
        str: A region-based family name with whitespace normalised.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.
        ValueError: If *city* is empty after stripping.

    Example:
        >>> region_surname('تهران')
        'تهرانی'
        >>> region_surname('اصفهان')
        'اصفهانی'
        >>> region_surname('تبریز')
        'تبریزی'
    """
    _require_random(rng)  # validate; the rule itself is deterministic
    city = city.strip()
    if not city:
        raise ValueError('city must be a non-empty place name')

    result = city if city.endswith(_TERMINATION) else city + _TERMINATION
    return normalize_name(result)


def occupation_surname(rng: Optional[random.Random] = None) -> str:
    """Return a random Persian occupational (trade) family name.

    Draws a whole surname from :data:`OCCUPATION_SURNAMES` (فلاح، قناد،
    خراط، …). Unlike :func:`compound_surname`, these are used as-is, not
    composed.

    Args:
        rng (random.Random, optional): Seeded RNG for reproducibility.

    Returns:
        str: An occupational family name from the pool.

    Raises:
        TypeError: If *rng* is not a ``random.Random``.

    Example:
        >>> import random
        >>> name = occupation_surname(rng=random.Random(3))
        >>> isinstance(name, str)
        True
    """
    generator = _require_random(rng)
    return generator.choice(OCCUPATION_SURNAMES)
