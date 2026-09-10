"""Dynamic extension of the embedded Persian name pools.

The shipped ``names.pkl`` historically under-represented common initials
(notably ``م`` and ``ف``). This module lets you:

1. Apply the built-in curated seed expansion (:func:`apply_seed_expansion`).
2. Merge your own names at runtime (:func:`extend_name_pools`).

Example:
    >>> from farsi_faker import FarsiFaker
    >>> from farsi_faker.extend import extend_name_pools
    >>> stats = extend_name_pools(male=['کوروش'], female=['لیلا'])
    >>> stats['male_added']
    1
    >>> 'کوروش' in FarsiFaker()._male_names
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .faker import FarsiFaker

__all__ = [
    'load_seed_expansion',
    'extend_name_pools',
    'apply_seed_expansion',
]

_SEED_MODULE = 'farsi_faker.data.seed_names'


def load_seed_expansion() -> Dict[str, Tuple[str, ...]]:
    """Load the curated built-in name expansion.

    Returns:
        Dict[str, Tuple[str, ...]]: Keys ``male_names``, ``female_names``,
        ``last_names`` with sorted unique tuples.

    Example:
        >>> seeds = load_seed_expansion()
        >>> 'محمد' in seeds['male_names']
        True
    """
    from .data.seed_names import FEMALE_NAMES, LAST_NAMES, MALE_NAMES

    return {
        'male_names': tuple(sorted(set(MALE_NAMES))),
        'female_names': tuple(sorted(set(FEMALE_NAMES))),
        'last_names': tuple(sorted(set(LAST_NAMES))),
    }


def _as_tuple(value: Optional[Sequence[str]], *, field: str) -> Tuple[str, ...]:
    """Validate and coerce an optional name sequence.

    Args:
        value (Sequence[str], optional): Caller-supplied names.
        field (str): Field name used in error messages.

    Returns:
        Tuple[str, ...]: Non-empty strings.

    Raises:
        TypeError: If *value* is provided but is not a sequence of strings.
    """
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f'{field} must be a sequence of strings, got {type(value)!r}')
    names: List[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f'{field} entries must be str, got {type(item)!r}')
        if item:
            names.append(item)
    return tuple(names)


def extend_name_pools(
    *,
    male: Optional[Sequence[str]] = None,
    female: Optional[Sequence[str]] = None,
    last: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Merge additional names into the process-wide FarsiFaker cache.

    New names are run through :func:`farsi_faker.cleaning.precision_repair`
    (male/female) or the last-name repair path, then merged, deduplicated,
    and sorted. The class-level cache is updated under a lock so all
    subsequently constructed instances see the expanded pools.

    Args:
        male (Sequence[str], optional): Extra male first names.
        female (Sequence[str], optional): Extra female first names.
        last (Sequence[str], optional): Extra family names.

    Returns:
        Dict[str, int]: Counts ``male_added``, ``female_added``,
        ``last_added`` (net new names after cleaning and dedupe).

    Raises:
        TypeError: If any argument is not a sequence of strings.

    Example:
        >>> from farsi_faker import FarsiFaker
        >>> from farsi_faker.extend import extend_name_pools
        >>> added = extend_name_pools(male=['کوروش'])
        >>> added['male_added']
        1
        >>> 'کوروش' in FarsiFaker()._male_names
        True
    """
    from .cleaning import precision_repair

    male_raw = _as_tuple(male, field='male')
    female_raw = _as_tuple(female, field='female')
    last_raw = _as_tuple(last, field='last')

    if not male_raw and not female_raw and not last_raw:
        return {'male_added': 0, 'female_added': 0, 'last_added': 0}

    # Ensure the embedded database is loaded before mutating the cache.
    FarsiFaker()

    def _repair_all(names: Sequence[str], pool_gender: str) -> List[str]:
        repaired: List[str] = []
        for name in names:
            cleaned = precision_repair(name, pool_gender=pool_gender)
            if cleaned:
                repaired.append(cleaned)
        return repaired

    male_new = _repair_all(male_raw, 'male')
    female_new = _repair_all(female_raw, 'female')
    # Last names: precision_repair with pool_gender last is allowed and
    # preserves multi-word compounds.
    last_new = _repair_all(last_raw, 'last')

    with FarsiFaker._data_lock:
        cache = FarsiFaker._data_cache
        if cache is None:  # pragma: no cover - constructor above fills cache
            raise RuntimeError('FarsiFaker cache failed to initialize')

        current_male = set(cache['male_names'])
        current_female = set(cache['female_names'])
        current_last = set(cache['last_names'])

        added_male = [n for n in male_new if n not in current_male]
        added_female = [n for n in female_new if n not in current_female]
        added_last = [n for n in last_new if n not in current_last]

        # If a name lands in both genders, keep it only on the side that
        # already holds it; new cross-gender adds prefer female (same as clean).
        if added_male and added_female:
            overlap = set(added_male) & set(added_female)
            added_male = [n for n in added_male if n not in overlap]

        new_male = sorted(current_male | set(added_male))
        new_female = sorted(current_female | set(added_female))
        new_last = sorted(current_last | set(added_last))

        # Male/female pools must stay disjoint.
        overlap_final = set(new_male) & set(new_female)
        if overlap_final:
            new_male = sorted(set(new_male) - overlap_final)

        FarsiFaker._data_cache = {
            'male_names': tuple(new_male),
            'female_names': tuple(new_female),
            'last_names': tuple(new_last),
        }

    # Refresh any already-constructed instance attributes is out of scope;
    # callers should construct FarsiFaker() after extending.
    return {
        'male_added': len(added_male),
        'female_added': len(added_female),
        'last_added': len(added_last),
    }


def apply_seed_expansion() -> Dict[str, int]:
    """Merge the built-in curated seed names into the live cache.

    Safe to call more than once; duplicates are ignored.

    Returns:
        Dict[str, int]: Counts of names actually added.

    Example:
        >>> from farsi_faker.extend import apply_seed_expansion
        >>> stats = apply_seed_expansion()
        >>> stats['male_added'] >= 0
        True
    """
    seeds = load_seed_expansion()
    return extend_name_pools(
        male=seeds['male_names'],
        female=seeds['female_names'],
        last=seeds['last_names'],
    )
