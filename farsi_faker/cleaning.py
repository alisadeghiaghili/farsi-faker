"""Name-pool cleaning utilities for Persian/Farsi name data.

Historical name datasets often contain OCR and segmentation artifacts
(for example ``'آ رمان'`` instead of ``'آرمان'``) and occasional gender-label
noise (honorifics such as ``بی بی`` appearing in the male pool).

This module exposes pure, testable helpers plus a single pipeline function
(:func:`clean_name_pools`) used when rebuilding the embedded database.

Example:
    >>> from farsi_faker.cleaning import join_ocr_splits, clean_name_pools
    >>> join_ocr_splits('آ رمان')
    'آرمان'
    >>> pools = clean_name_pools({
    ...     'male_names': ['آ رمان', 'آرمان', 'بی بی رضا'],
    ...     'female_names': ['فاطمه'],
    ...     'last_names': ['احمدی'],
    ... })
    >>> pools['male_names']
    ['آرمان']
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    'normalize_name',
    'has_singleton_token',
    'join_ocr_splits',
    'drop_from_pool',
    'dedupe_preserve_order',
    'clean_name_pools',
]

REQUIRED_POOL_KEYS = ('male_names', 'female_names', 'last_names')

# Leading token that is almost always truncated آقا/اقا (OCR dropped final ی/الف).
_AGHA_ABBREVIATIONS = frozenset({'آق', 'اق'})

# Female honorifics / titles that should never appear in a male first-name pool.
_FEMALE_HONORIFICS = (
    'بی بی',
    'بی‌بی',
    'بیگم',
    'خانم',
    'بانو',
)

# Male titles that should not appear in a female first-name pool when leading.
# Note: حاج / حاجی are unisex and must stay allowed in both pools.
_MALE_TITLE_PREFIXES = (
    'آقا ',
    'اقا ',
    'میرزا ',
)


def normalize_name(name: Optional[str]) -> str:
    """Normalize whitespace on a name string.

    Collapses internal runs of whitespace, strips ends, and leaves ZWNJ
    (U+200C) untouched so Persian orthography is preserved.

    Args:
        name (str, optional): Raw name value. ``None`` and empty inputs
            become ``''``.

    Returns:
        str: Normalized name, or empty string when input is blank.

    Example:
        >>> normalize_name('  علی   احمدی ')
        'علی احمدی'
        >>> normalize_name(None)
        ''
        >>> normalize_name('می‌رود')
        'می‌رود'
    """
    if name is None:
        return ''
    return ' '.join(str(name).split())


def _tokens(name: str) -> List[str]:
    """Split a name into non-empty whitespace tokens.

    Args:
        name (str): Name already normalized (or raw).

    Returns:
        List[str]: Tokens; empty list when the name is blank.
    """
    return [token for token in name.split() if token]


def has_singleton_token(name: str) -> bool:
    """Return True when any whitespace token is a single character.

    A one-character token in a Persian first or family name is a strong OCR
    artifact signal (``'آ رمان'``, ``'د ا و د'``).

    Args:
        name (str): Name to inspect.

    Returns:
        bool: True if at least one token has length 1.

    Example:
        >>> has_singleton_token('آ رمان')
        True
        >>> has_singleton_token('محمد رضا')
        False
    """
    return any(len(token) == 1 for token in _tokens(name))


def join_ocr_splits(name: str, *, allow_short_head_join: bool = True) -> str:
    """Repair common OCR/segmentation splits in a Persian name.

    Rules (applied in order):

    1. Normalize whitespace.
    2. If any token has length 1, join all tokens (``'آ رمان'`` → ``'آرمان'``).
    3. When *allow_short_head_join* is True and there are exactly two tokens
       with a length-2 head, treat it as a mid-word split and join
       (``'آر مان'`` → ``'آرمان'``), unless the head is a truncated
       ``آقا``/``اقا`` prefix — those expand and keep a space
       (``'آق محمد'`` → ``'آقا محمد'``).
    4. Otherwise return the normalized name unchanged so legitimate compounds
       (``'محمد رضا'``, ``'آب روشن'``) are preserved.

    Family names should call this with ``allow_short_head_join=False``.

    Args:
        name (str): Raw or normalized name.
        allow_short_head_join (bool, optional): Apply the two-token short-head
            join rule. Defaults to True. Pass False for family names.

    Returns:
        str: Repaired name.

    Example:
        >>> join_ocr_splits('ا میر')
        'امیر'
        >>> join_ocr_splits('با با')
        'بابا'
        >>> join_ocr_splits('محمد رضا')
        'محمد رضا'
        >>> join_ocr_splits('آق محمد')
        'آقا محمد'
        >>> join_ocr_splits('آب روشن', allow_short_head_join=False)
        'آب روشن'
    """
    normalized = normalize_name(name)
    if not normalized:
        return ''

    tokens = _tokens(normalized)
    if len(tokens) <= 1:
        return normalized

    if any(len(token) == 1 for token in tokens):
        return ''.join(tokens)

    if allow_short_head_join and len(tokens) == 2 and len(tokens[0]) == 2:
        head, tail = tokens
        if head in _AGHA_ABBREVIATIONS:
            expanded = 'آقا' if head == 'آق' else 'اقا'
            return f'{expanded} {tail}'
        return head + tail

    return normalized


def drop_from_pool(name: str, pool_gender: str) -> bool:
    """Decide whether a name should be removed from a gendered pool.

    Args:
        name (str): Name under consideration (already normalized).
        pool_gender (str): ``'male'``, ``'female'``, or ``'last'``.

    Returns:
        bool: True when the name must be dropped from that pool.

    Example:
        >>> drop_from_pool('بی بی مریم', pool_gender='male')
        True
        >>> drop_from_pool('علی', pool_gender='male')
        False
        >>> drop_from_pool('آب روشن', pool_gender='last')
        False
    """
    if pool_gender == 'last':
        return False

    text = normalize_name(name)

    if pool_gender == 'male':
        return any(marker in text for marker in _FEMALE_HONORIFICS)

    if pool_gender == 'female':
        return any(text.startswith(prefix) for prefix in _MALE_TITLE_PREFIXES)

    return False


def dedupe_preserve_order(names: Iterable[str]) -> List[str]:
    """Remove duplicates while keeping the first occurrence order.

    Args:
        names (Iterable[str]): Input names.

    Returns:
        List[str]: Deduplicated names in original relative order.

    Example:
        >>> dedupe_preserve_order(['الف', 'ب', 'الف'])
        ['الف', 'ب']
    """
    seen = set()
    result: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _clean_first_name_pool(
    names: Sequence[str],
    pool_gender: str,
    cross_gender: set[str],
) -> List[str]:
    """Repair, filter, and dedupe a first-name pool.

    Args:
        names (Sequence[str]): Raw first names.
        pool_gender (str): ``'male'`` or ``'female'``.
        cross_gender (set[str]): Names retained by the opposite gender pool
            after cleaning; used to drop true overlaps from this side when the
            name is honorific-contaminated.

    Returns:
        List[str]: Cleaned, sorted, unique names.
    """
    repaired: List[str] = []
    for raw in names:
        name = join_ocr_splits(raw)
        if not name:
            continue
        if drop_from_pool(name, pool_gender=pool_gender):
            continue
        if pool_gender == 'male' and name in cross_gender:
            # Prefer keeping a contested name in the female pool when both
            # sides produced the same repaired token; male side drops it.
            continue
        repaired.append(name)

    return sorted(set(repaired))


def clean_name_pools(raw: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Clean male, female, and family-name pools in one pass.

    Pipeline:

    1. Validate required keys.
    2. Repair OCR splits (see :func:`join_ocr_splits`).
    3. Drop gender-label noise (see :func:`drop_from_pool`).
    4. Resolve residual male/female overlaps by keeping the name on the
       female side only.
    5. Deduplicate and return sorted lists.

    Family names keep multi-word compounds (``'آب روشن'``) because those are
    legitimate Persian surnames.

    Args:
        raw (Mapping[str, Sequence[str]]): Mapping with keys
            ``male_names``, ``female_names``, and ``last_names``.

    Returns:
        Dict[str, List[str]]: Mapping with the same three keys and cleaned,
        sorted, unique name lists.

    Raises:
        KeyError: If any required pool key is missing.

    Example:
        >>> clean_name_pools({
        ...     'male_names': ['آ رمان', 'آرمان', 'بی بی رضا'],
        ...     'female_names': ['فاطمه'],
        ...     'last_names': ['احمدی', 'احمدی'],
        ... })
        {'male_names': ['آرمان'], 'female_names': ['فاطمه'], 'last_names': ['احمدی']}
    """
    missing = [key for key in REQUIRED_POOL_KEYS if key not in raw]
    if missing:
        raise KeyError(f'clean_name_pools missing keys: {missing}')

    # Female first so contested male/female repairs keep the female entry.
    female_repaired = [
        name
        for name in (join_ocr_splits(item) for item in raw['female_names'])
        if name and not drop_from_pool(name, pool_gender='female')
    ]
    female_set = set(female_repaired)

    male_clean = _clean_first_name_pool(
        raw['male_names'],
        pool_gender='male',
        cross_gender=female_set,
    )
    female_clean = sorted(female_set)

    last_clean = sorted(
        {
            name
            for name in (
                join_ocr_splits(item, allow_short_head_join=False)
                for item in raw['last_names']
            )
            if name
        }
    )

    return {
        'male_names': male_clean,
        'female_names': female_clean,
        'last_names': last_clean,
    }
