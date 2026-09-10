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

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = [
    'normalize_name',
    'normalize_zwnj',
    'apply_zwnj_policy',
    'has_singleton_token',
    'join_ocr_splits',
    'join_abdol_family',
    'is_truncated_name',
    'is_too_short',
    'precision_repair',
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


ZWNJ = '‌'

# Space-separated compounds that take a ZWNJ (نیم‌فاصله) in modern orthography.
# Keep this list conservative: only high-confidence lexical pairs, not names.
_ZWNJ_COMPOUNDS = frozenset(
    {
        ('می', 'خواهد'),
        ('می', 'رود'),
        ('می', 'آید'),
        ('می', 'دهد'),
        ('می', 'گوید'),
        ('می', 'بیند'),
        ('می', 'شود'),
        ('می', 'تواند'),
        ('خانه', 'دار'),
        ('کتاب', 'خانه'),
        ('مدرسه', 'رو'),
    }
)


def normalize_zwnj(name: Optional[str]) -> str:
    """Clean ZWNJ placement around spaces and at string edges.

    Removes spaces that sit next to a ZWNJ and strips leading/trailing ZWNJ.
    Internal ZWNJ characters are preserved.

    Args:
        name (str, optional): Raw or normalized name.

    Returns:
        str: ZWNJ-hygienic string; empty when input is blank.

    Example:
        >>> normalize_zwnj('می‌ رود')
        'می‌رود'
        >>> normalize_zwnj('‌علی‌')
        'علی'
        >>> normalize_zwnj('می‌رود')
        'می‌رود'
    """
    if name is None:
        return ''
    text = str(name)
    # Drop spaces that abut ZWNJ.
    text = re.sub(rf'{ZWNJ}\s+', ZWNJ, text)
    text = re.sub(rf'\s+{ZWNJ}', ZWNJ, text)
    # Strip ZWNJ at edges.
    text = text.strip(ZWNJ)
    return text


def apply_zwnj_policy(name: str) -> str:
    """Rewrite known space-separated compounds using ZWNJ.

    Only pairs listed in the internal compound set are rewritten. Person
    names such as ``محمد رضا`` are left unchanged.

    Args:
        name (str): Name already whitespace-normalized.

    Returns:
        str: Name with selected compounds joined by ZWNJ.

    Example:
        >>> apply_zwnj_policy('می خواهد')
        'می‌خواهد'
        >>> apply_zwnj_policy('محمد رضا')
        'محمد رضا'
    """
    normalized = normalize_zwnj(normalize_name(name))
    if not normalized:
        return ''

    tokens = _tokens(normalized)
    if len(tokens) < 2:
        return normalized

    result: List[str] = [tokens[0]]
    index = 1
    while index < len(tokens):
        pair = (result[-1], tokens[index])
        if pair in _ZWNJ_COMPOUNDS:
            result[-1] = result[-1] + ZWNJ + tokens[index]
        else:
            result.append(tokens[index])
        index += 1
    return ' '.join(result)


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


# Prefixes that in modern Persian orthography are almost always one word
# with what follows (عبدالله, عبدالرحمن, ...).
_ABDOL_PREFIXES = ('عبد', 'عب')

# Bare prefix tokens that are incomplete when they stand alone.
_INCOMPLETE_STANDALONE = frozenset({'عبد', 'عب', 'ال', 'آق', 'اق'})

# Trailing token that indicates a truncated OCR remainder.
_TRUNCATED_TOKEN = 'ال'


def join_abdol_family(name: str) -> str:
    """Glue ``عبد`` / ``عب`` family names into a single token.

    Iranian sources often split ``عبدالله`` as ``عبد الله`` or worse OCR
    fragments such as ``عب الر ضا``.

    Args:
        name (str): Name already whitespace-normalized (or raw).

    Returns:
        str: Repaired name. Unchanged when the Abdol prefix rule does not
        apply.

    Example:
        >>> join_abdol_family('عبد الله')
        'عبدالله'
        >>> join_abdol_family('عبد الر ضا')
        'عبدالرضا'
        >>> join_abdol_family('امیر عبد الله')
        'امیر عبدالله'
        >>> join_abdol_family('علی')
        'علی'
    """
    normalized = normalize_name(name)
    if not normalized:
        return ''

    tokens = _tokens(normalized)
    if not tokens:
        return ''

    # Find the first Abdol prefix token and glue it with everything after.
    for index, token in enumerate(tokens):
        if token in _ABDOL_PREFIXES:
            head = tokens[:index]
            glued = token + ''.join(tokens[index + 1 :])
            if head:
                return f"{' '.join(head)} {glued}"
            return glued

    return normalized


def is_truncated_name(name: str) -> bool:
    """Return True when the name ends with a bare truncated ``ال`` token.

    Args:
        name (str): Name to inspect.

    Returns:
        bool: True if the final token is exactly ``ال``.

    Example:
        >>> is_truncated_name('اسما ال')
        True
        >>> is_truncated_name('عبدالله')
        False
        >>> is_truncated_name('محمد')
        False
    """
    tokens = _tokens(normalize_name(name))
    return bool(tokens) and tokens[-1] == _TRUNCATED_TOKEN


def is_too_short(name: str, *, min_letters: int = 3) -> bool:
    """Return True when the name is shorter than *min_letters* letters.

    Whitespace is ignored. Used to drop OCR fragments such as ``'آبث'``.

    Args:
        name (str): Name to inspect.
        min_letters (int, optional): Minimum letter count. Defaults to 3.

    Returns:
        bool: True when the compacted name is shorter than the threshold.

    Example:
        >>> is_too_short('آبث')
        True
        >>> is_too_short('آرش')
        False
        >>> is_too_short('آ ر')
        True
    """
    compact = normalize_name(name).replace(' ', '')
    return len(compact) < min_letters


def precision_repair(name: str, *, pool_gender: str) -> Optional[str]:
    """Apply the precision repair pipeline to a single name.

    Order:

    1. :func:`join_ocr_splits` (singleton / short-head splits)
    2. :func:`join_abdol_family`
    3. :func:`normalize_zwnj`
    4. Drop if :func:`is_truncated_name`
    5. Drop if :func:`is_too_short`
    6. Drop if :func:`drop_from_pool` flags gender-label noise

    Args:
        name (str): Raw name.
        pool_gender (str): ``'male'``, ``'female'``, or ``'last'``.

    Returns:
        Optional[str]: Repaired name, or ``None`` when the name must be
        discarded.

    Example:
        >>> precision_repair('عبد الر ضا', pool_gender='male')
        'عبدالرضا'
        >>> precision_repair('اسما ال', pool_gender='female') is None
        True
        >>> precision_repair('عبد', pool_gender='male') is None
        True
        >>> precision_repair('آرش', pool_gender='male')
        'آرش'
    """
    repaired = normalize_zwnj(join_abdol_family(join_ocr_splits(name)))
    if not repaired:
        return None
    if repaired in _INCOMPLETE_STANDALONE:
        return None
    if is_truncated_name(repaired):
        return None
    if is_too_short(repaired):
        return None
    if drop_from_pool(repaired, pool_gender=pool_gender):
        return None
    return repaired


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
        name = precision_repair(raw, pool_gender=pool_gender)
        if not name:
            continue
        if pool_gender == 'male' and name in cross_gender:
            continue
        repaired.append(name)

    return sorted(set(repaired))


def clean_name_pools(raw: Mapping[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Clean male, female, and family-name pools in one pass.

    Pipeline:

    1. Validate required keys.
    2. Precision-repair each name (OCR joins, Abdol glue, truncation filter).
    3. Drop gender-label noise and tiny OCR fragments.
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
        ...     'male_names': ['آ رمان', 'آرمان', 'بی بی رضا', 'عبد الر ضا'],
        ...     'female_names': ['فاطمه', 'اسما ال'],
        ...     'last_names': ['احمدی', 'احمدی'],
        ... })
        {'male_names': ['آرمان', 'عبدالرضا'], 'female_names': ['فاطمه'], 'last_names': ['احمدی']}
    """
    missing = [key for key in REQUIRED_POOL_KEYS if key not in raw]
    if missing:
        raise KeyError(f'clean_name_pools missing keys: {missing}')

    female_repaired = []
    for item in raw['female_names']:
        name = precision_repair(item, pool_gender='female')
        if name:
            female_repaired.append(name)
    female_set = set(female_repaired)

    male_clean = _clean_first_name_pool(
        raw['male_names'],
        pool_gender='male',
        cross_gender=female_set,
    )
    female_clean = sorted(female_set)

    last_clean = set()
    for item in raw['last_names']:
        name = join_abdol_family(join_ocr_splits(item, allow_short_head_join=False))
        if not name or name in _INCOMPLETE_STANDALONE:
            continue
        if is_truncated_name(name) or is_too_short(name):
            continue
        last_clean.add(name)

    return {
        'male_names': male_clean,
        'female_names': female_clean,
        'last_names': sorted(last_clean),
    }
