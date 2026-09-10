"""Rebuild the embedded names.pkl by applying farsi_faker.cleaning.

Usage (from project root):
    python scripts/rebuild_names_pkl.py
    python scripts/rebuild_names_pkl.py --dry-run
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from farsi_faker.cleaning import clean_name_pools, has_singleton_token

DEFAULT_PKL = Path(__file__).resolve().parent.parent / 'farsi_faker' / 'data' / 'names.pkl'


def _stats(pools: dict) -> dict:
    male = pools['male_names']
    female = pools['female_names']
    last = pools['last_names']
    singleton = sum(
        1
        for pool in (male, female, last)
        for name in pool
        if has_singleton_token(name)
    )
    return {
        'male': len(male),
        'female': len(female),
        'last': len(last),
        'singleton_remaining': singleton,
        'overlap': len(set(male) & set(female)),
    }


def rebuild(pkl_path: Path, *, dry_run: bool = False) -> dict:
    """Load, clean, and optionally rewrite names.pkl.

    Args:
        pkl_path (Path): Path to the pickle database.
        dry_run (bool): When True, do not write the file.

    Returns:
        dict: Before/after statistics.
    """
    with pkl_path.open('rb') as handle:
        raw = pickle.load(handle)

    before = _stats(raw)
    cleaned = clean_name_pools(raw)
    after = _stats(cleaned)

    print('Before:', before)
    print('After: ', after)
    print(
        'Removed: male={m} female={f} last={l}'.format(
            m=before['male'] - after['male'],
            f=before['female'] - after['female'],
            l=before['last'] - after['last'],
        )
    )

    if not dry_run:
        with pkl_path.open('wb') as handle:
            pickle.dump(cleaned, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Wrote {pkl_path} ({pkl_path.stat().st_size / 1024:.2f} KB)')

    return {'before': before, 'after': after}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkl', type=Path, default=DEFAULT_PKL)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    if not args.pkl.exists():
        print(f'Not found: {args.pkl}', file=sys.stderr)
        return 1

    rebuild(args.pkl, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
