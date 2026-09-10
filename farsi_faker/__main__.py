"""Command-line interface for farsi-faker.

Usage:
    python -m farsi_faker --count 5 --gender male
    python -m farsi_faker --count 100 --format csv > people.csv
    python -m farsi_faker --profile --count 3 --format json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from ._version import __version__
from .faker import FarsiFaker


def _build_records(
    count: int,
    gender: Optional[str],
    profile: bool,
    seed: Optional[int],
) -> List[Dict[str, Any]]:
    """Generate person records for the CLI.

    Args:
        count (int): Number of records.
        gender (str, optional): Gender filter.
        profile (bool): Include contact fields when True.
        seed (int, optional): RNG seed.

    Returns:
        List[Dict[str, Any]]: Generated records.
    """
    faker = FarsiFaker(seed=seed)
    if profile:
        return [faker.profile(gender) for _ in range(count)]
    return [faker.full_name(gender) for _ in range(count)]


def _emit(records: Sequence[Dict[str, Any]], fmt: str) -> None:
    """Write records to stdout as JSON or CSV.

    Args:
        records (Sequence[Dict[str, Any]]): Records to emit.
        fmt (str): ``'json'`` or ``'csv'``.
    """
    if not records:
        return

    if fmt == 'json':
        json.dump(list(records), sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write('\n')
        return

    fieldnames = list(records[0].keys())
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog='farsi-faker',
        description='Generate synthetic Persian names and profile fields.',
    )
    parser.add_argument(
        '-n',
        '--count',
        type=int,
        default=1,
        help='number of records to generate (default: 1)',
    )
    parser.add_argument(
        '-g',
        '--gender',
        choices=['male', 'female'],
        default=None,
        help='restrict gender',
    )
    parser.add_argument(
        '-f',
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='output format (default: json)',
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='include national_id, mobile, email, postal_code',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed for reproducible output',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv (Sequence[str], optional): Argument vector. Defaults to
            ``sys.argv[1:]``.

    Returns:
        int: Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error('count must be a positive integer')

    records = _build_records(
        count=args.count,
        gender=args.gender,
        profile=args.profile,
        seed=args.seed,
    )
    _emit(records, args.format)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
