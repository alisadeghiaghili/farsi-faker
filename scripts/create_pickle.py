"""Convert CSV names data to optimized pickle format for farsi-faker.

This script processes the Iranian names CSV files and creates an optimized
pickle file for fast loading in the faker. The pickle format provides:
- Faster loading times (10-100x faster than CSV)
- Smaller file size
- No runtime dependencies on pandas for consumers of the package

Usage:
    # As script (from project root):
    python scripts/create_pickle.py

    # Optional explicit project root:
    python scripts/create_pickle.py /path/to/farsi-faker
"""

import pickle
import re
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

# Persian block U+0600-U+06FF + ZWNJ (U+200C) + whitespace
_PERSIAN_NAME_PATTERN = re.compile(r'^[؀-ۿ‌\s]+$')


def _require_pandas():
    """Import pandas on demand.

    Returns:
        module: The pandas module.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd_mod
    except ImportError as exc:
        raise ImportError(
            'pandas is required to build names.pkl. Install with: pip install pandas'
        ) from exc
    return pd_mod


def clean_name(name: object) -> str:
    """Clean and normalize a name string.

    Args:
        name (object): Raw name value from CSV; may be None or NaN.

    Returns:
        str: Cleaned name with collapsed whitespace, or empty string if invalid.
    """
    if name is None:
        return ''

    if not isinstance(name, str):
        pd = _require_pandas()
        if pd.isna(name):
            return ''

    return ' '.join(str(name).split())


def is_valid_persian(text: str) -> bool:
    """Check if text contains only Persian characters, ZWNJ, and spaces.

    Args:
        text (str): Text to validate.

    Returns:
        bool: True when non-empty and matching the Persian name charset.

    Example:
        >>> is_valid_persian('علی')
        True
        >>> is_valid_persian('Ali')
        False
        >>> is_valid_persian('می‌رود')
        True
    """
    if not text:
        return False
    return bool(_PERSIAN_NAME_PATTERN.match(text))


def load_first_names(csv_path: Path) -> Tuple[Set[str], Set[str]]:
    """Load and process first names from CSV with gender classification.

    Args:
        csv_path (Path): Path to iranianNamesDataset.csv.

    Returns:
        Tuple[Set[str], Set[str]]: ``(male_names, female_names)``.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ImportError: If pandas is unavailable.
    """
    pd = _require_pandas()
    print(f'Loading first names from: {csv_path.name}')

    if not csv_path.exists():
        raise FileNotFoundError(f'File not found: {csv_path}')

    frame = pd.read_csv(csv_path, encoding='utf-8')

    male_names: Set[str] = set()
    female_names: Set[str] = set()
    skipped = 0

    for _, row in frame.iterrows():
        name = clean_name(row.iloc[0])

        if not is_valid_persian(name):
            skipped += 1
            continue

        if len(row) > 1:
            gender = str(row.iloc[1]).strip().upper()
            if gender == 'M':
                male_names.add(name)
            elif gender == 'F':
                female_names.add(name)
            else:
                skipped += 1
        else:
            skipped += 1

    print(f'   Male names: {len(male_names):,}')
    print(f'   Female names: {len(female_names):,}')
    print(f'   Skipped: {skipped:,}')

    return male_names, female_names


def load_last_names(csv_path: Path) -> Set[str]:
    """Load and process last names from CSV.

    Args:
        csv_path (Path): Path to iranian-surname-frequencies.csv.

    Returns:
        Set[str]: Unique family names.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ImportError: If pandas is unavailable.
    """
    pd = _require_pandas()
    print(f'Loading last names from: {csv_path.name}')

    if not csv_path.exists():
        raise FileNotFoundError(f'File not found: {csv_path}')

    frame = pd.read_csv(csv_path, encoding='utf-8')

    last_names: Set[str] = set()
    skipped = 0

    for _, row in frame.iterrows():
        name = clean_name(row.iloc[0])

        if is_valid_persian(name):
            last_names.add(name)
        else:
            skipped += 1

    print(f'   Last names: {len(last_names):,}')
    print(f'   Skipped: {skipped:,}')

    return last_names


def create_pickle_data(
    first_names_csv: Path,
    last_names_csv: Path,
    output_pickle: Path,
) -> None:
    """Create optimized pickle file from CSV sources.

    Args:
        first_names_csv (Path): Path to first names CSV.
        last_names_csv (Path): Path to last names CSV.
        output_pickle (Path): Destination pickle path.

    Raises:
        ValueError: If any name pool is smaller than 100 entries.
    """
    print('=' * 60)
    print('Creating optimized names database for farsi-faker')
    print('=' * 60)

    male_names, female_names = load_first_names(first_names_csv)
    last_names = load_last_names(last_names_csv)

    if len(male_names) < 100:
        raise ValueError(
            f'Insufficient male names: {len(male_names)} (need at least 100)'
        )
    if len(female_names) < 100:
        raise ValueError(
            f'Insufficient female names: {len(female_names)} (need at least 100)'
        )
    if len(last_names) < 100:
        raise ValueError(
            f'Insufficient last names: {len(last_names)} (need at least 100)'
        )

    data = {
        'male_names': sorted(male_names),
        'female_names': sorted(female_names),
        'last_names': sorted(last_names),
    }

    print(f'Saving to: {output_pickle}')
    output_pickle.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pickle, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_kb = output_pickle.stat().st_size / 1024
    print(f'   File size: {file_size_kb:.2f} KB')

    total_names = sum(len(values) for values in data.values())
    combinations = (
        len(data['male_names']) + len(data['female_names'])
    ) * len(data['last_names'])

    print('=' * 60)
    print('Database statistics')
    print('=' * 60)
    print(f'Male first names:      {len(data["male_names"]):>10,}')
    print(f'Female first names:    {len(data["female_names"]):>10,}')
    print(f'Family names:          {len(data["last_names"]):>10,}')
    print('-' * 60)
    print(f'Total unique names:    {total_names:>10,}')
    print(f'Possible combinations: {combinations:>10,}')
    print('=' * 60)
    print('Pickle file created successfully.')


def get_script_dir(manual_path: Optional[str] = None) -> Path:
    """Resolve the project root directory.

    Args:
        manual_path (str, optional): Explicit project root.

    Returns:
        Path: Project root directory.

    Raises:
        FileNotFoundError: If a manual path is provided and missing.
    """
    if manual_path:
        path = Path(manual_path)
        if path.exists():
            return path
        raise FileNotFoundError(f'Manual path does not exist: {path}')

    try:
        script_path = Path(__file__).resolve()
        if script_path.parent.name == 'scripts':
            return script_path.parent.parent
        return script_path.parent
    except NameError:
        current = Path.cwd()
        if (current / 'setup.py').exists() or (current / 'farsi_faker').exists():
            return current
        if (current.parent / 'setup.py').exists() or (
            current.parent / 'farsi_faker'
        ).exists():
            return current.parent
        print('Warning: could not auto-detect project root.')
        print(f'   Current directory: {current}')
        print('   Provide manual_path if this is incorrect.')
        return current


def main(project_root: Optional[str] = None) -> None:
    """Run the pickle build pipeline.

    Args:
        project_root (str, optional): Explicit project root path.
    """
    script_dir = get_script_dir(project_root)
    data_sources_dir = script_dir / 'data_sources'
    first_names_csv = data_sources_dir / 'iranianNamesDataset.csv'
    last_names_csv = data_sources_dir / 'iranian-surname-frequencies.csv'
    output_pickle = script_dir / 'farsi_faker' / 'data' / 'names.pkl'

    print('=' * 60)
    print('Path configuration')
    print('=' * 60)
    print(f'Project root:    {script_dir}')
    print(f'Data sources:    {data_sources_dir}')
    print(f'Output pickle:   {output_pickle}')
    print('=' * 60)

    if not data_sources_dir.exists():
        print(f'Error: data_sources directory not found: {data_sources_dir}')
        print("Tip: main(project_root='C:/path/to/farsi-faker')")
        sys.exit(1)

    try:
        create_pickle_data(first_names_csv, last_names_csv, output_pickle)
    except FileNotFoundError as exc:
        print(f'Error: {exc}')
        print('Expected layout:')
        print(f'  {script_dir}/')
        print('  ├── data_sources/')
        print('  │   ├── iranianNamesDataset.csv')
        print('  │   └── iranian-surname-frequencies.csv')
        print('  └── farsi_faker/data/')
        sys.exit(1)
    except Exception as exc:
        print(f'Error: {exc}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(project_root=sys.argv[1])
    else:
        main()
