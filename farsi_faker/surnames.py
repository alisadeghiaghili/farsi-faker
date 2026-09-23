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
    'compound_surname',
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
