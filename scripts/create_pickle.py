"""Convert CSV names data to optimized pickle format for farsi-faker.

This script processes the Iranian names CSV files and creates an optimized
pickle file for fast loading in the faker. The pickle format provides:
- Faster loading times (10-100x faster than CSV)
- Smaller file size
- No runtime dependencies on pandas

Usage:
    # As script (from project root):
    python scripts/create_pickle.py

    # In notebook (specify project_root):
    project_root = Path('C:/path/to/farsi-faker')  # اصلاح کن!
    create_pickle_data(...)
"""

import pickle
import pandas as pd
from pathlib import Path
from typing import Set, Tuple, Optional
import re
import sys
import os


def clean_name(name: str) -> str:
    """Clean and normalize a name string.

    Args:
        name: Raw name string from CSV

    Returns:
        Cleaned and normalized name, or empty string if invalid
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()
    # Remove extra whitespace
    name = " ".join(name.split())
    return name


def is_valid_persian(text: str) -> bool:
    """Check if text contains only Persian characters and spaces.

    Args:
        text: Text to validate

    Returns:
        True if text is valid Persian, False otherwise
    """
    if not text:
        return False

    # Persian Unicode range: U+0600 to U+06FF
    persian_pattern = re.compile(r'^[\u0600-\u06FF\s]+$')
    return bool(persian_pattern.match(text))


def load_first_names(csv_path: Path) -> Tuple[Set[str], Set[str]]:
    """Load and process first names from CSV with gender classification.

    Args:
        csv_path: Path to iranianNamesDataset.csv

    Returns:
        Tuple of (male_names, female_names) as sets
    """
    print(f"📖 Loading first names from: {csv_path.name}")

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8')

    male_names: Set[str] = set()
    female_names: Set[str] = set()
    skipped = 0

    for _, row in df.iterrows():
        name = clean_name(row.iloc[0])

        if not is_valid_persian(name):
            skipped += 1
            continue

        # Check gender column (assumes second column is gender: M/F)
        if len(row) > 1:
            gender = str(row.iloc[1]).strip().upper()
            if gender == 'M':
                male_names.add(name)
            elif gender == 'F':
                female_names.add(name)
            else:
                # Unknown gender, skip
                skipped += 1
        else:
            # No gender column, skip
            skipped += 1

    print(f"   ✓ Male names: {len(male_names):,}")
    print(f"   ✓ Female names: {len(female_names):,}")
    print(f"   ⚠ Skipped: {skipped:,}")

    return male_names, female_names


def load_last_names(csv_path: Path) -> Set[str]:
    """Load and process last names from CSV.

    Args:
        csv_path: Path to iranian-surname-frequencies.csv

    Returns:
        Set of family names
    """
    print(f"📖 Loading last names from: {csv_path.name}")

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding='utf-8')

    last_names: Set[str] = set()
    skipped = 0

    for _, row in df.iterrows():
        name = clean_name(row.iloc[0])

        if is_valid_persian(name):
            last_names.add(name)
        else:
            skipped += 1

    print(f"   ✓ Last names: {len(last_names):,}")
    print(f"   ⚠ Skipped: {skipped:,}")

    return last_names


def create_pickle_data(
    first_names_csv: Path,
    last_names_csv: Path,
    output_pickle: Path
) -> None:
    """Create optimized pickle file from CSV sources.

    Args:
        first_names_csv: Path to first names CSV
        last_names_csv: Path to last names CSV
        output_pickle: Path for output pickle file
    """
    print("\n" + "="*60)
    print("🚀 Creating optimized names database for farsi-faker")
    print("="*60 + "\n")

    # Load data
    male_names, female_names = load_first_names(first_names_csv)
    last_names = load_last_names(last_names_csv)

    # Validate minimum requirements
    if len(male_names) < 100:
        raise ValueError(f"Insufficient male names: {len(male_names)} (need at least 100)")
    if len(female_names) < 100:
        raise ValueError(f"Insufficient female names: {len(female_names)} (need at least 100)")
    if len(last_names) < 100:
        raise ValueError(f"Insufficient last names: {len(last_names)} (need at least 100)")

    # Convert to sorted lists for consistent ordering and better compression
    data = {
        'male_names': sorted(list(male_names)),
        'female_names': sorted(list(female_names)),
        'last_names': sorted(list(last_names))
    }

    # Save to pickle with highest protocol for best performance
    print(f"\n💾 Saving to: {output_pickle}")
    output_pickle.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pickle, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Verify and report
    file_size_kb = output_pickle.stat().st_size / 1024
    print(f"   ✓ File size: {file_size_kb:.2f} KB")

    # Calculate statistics
    total_names = sum(len(v) for v in data.values())
    combinations = (len(data['male_names']) + len(data['female_names'])) * len(data['last_names'])

    print("\n" + "="*60)
    print("📊 Database Statistics")
    print("="*60)
    print(f"Male first names:      {len(data['male_names']):>10,}")
    print(f"Female first names:    {len(data['female_names']):>10,}")
    print(f"Family names:          {len(data['last_names']):>10,}")
    print(f"{'─'*60}")
    print(f"Total unique names:    {total_names:>10,}")
    print(f"Possible combinations: {combinations:>10,}")
    print("="*60)

    print("\n✅ Pickle file created successfully!")
    print(f"📦 Ready for packaging in farsi-faker\n")


def get_script_dir(manual_path: Optional[str] = None) -> Path:
    """Get the project root directory.

    Args:
        manual_path: Manual path to project root (for notebook usage)

    Returns:
        Path to project root directory
    """
    # If manual path provided, use it
    if manual_path:
        path = Path(manual_path)
        if path.exists():
            return path
        else:
            raise FileNotFoundError(f"Manual path does not exist: {path}")

    # Try to get __file__ (works when run as script)
    try:
        script_path = Path(__file__).resolve()
        # If script is in scripts/ folder
        if script_path.parent.name == 'scripts':
            return script_path.parent.parent
        # If script is in project root
        else:
            return script_path.parent
    except NameError:
        # Running in notebook/REPL
        # Try to detect project root by looking for setup.py or farsi_faker folder
        current = Path.cwd()

        # Check current directory
        if (current / 'setup.py').exists() or (current / 'farsi_faker').exists():
            return current

        # Check parent directory
        if (current.parent / 'setup.py').exists() or (current.parent / 'farsi_faker').exists():
            return current.parent

        # If nothing found, return current directory
        print(f"⚠️  Warning: Could not auto-detect project root.")
        print(f"   Current directory: {current}")
        print(f"   Please provide manual_path parameter if this is incorrect.")
        return current


def main(project_root: Optional[str] = None):
    """Main function to run the script.

    Args:
        project_root: Optional manual path to project root (for notebook usage)
    """
    # Define paths
    script_dir = get_script_dir(project_root)
    data_sources_dir = script_dir / 'data_sources'
    first_names_csv = data_sources_dir / 'iranianNamesDataset.csv'
    last_names_csv = data_sources_dir / 'iranian-surname-frequencies.csv'
    output_pickle = script_dir / 'farsi_faker' / 'data' / 'names.pkl'

    print("="*60)
    print("📂 Path Configuration")
    print("="*60)
    print(f"Project root:    {script_dir}")
    print(f"Data sources:    {data_sources_dir}")
    print(f"Output pickle:   {output_pickle}")
    print("="*60)

    # Check if paths exist
    if not data_sources_dir.exists():
        print(f"\n❌ Error: data_sources directory not found!")
        print(f"   Expected: {data_sources_dir}")
        print(f"\n💡 Tip: Run this script from project root, or provide manual path:")
        print(f"   main(project_root='C:/path/to/farsi-faker')")
        sys.exit(1)

    try:
        create_pickle_data(first_names_csv, last_names_csv, output_pickle)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Expected directory structure:")
        print(f"   {script_dir}/")
        print(f"   ├── data_sources/")
        print(f"   │   ├── iranianNamesDataset.csv")
        print(f"   │   └── iranian-surname-frequencies.csv")
        print(f"   └── farsi_faker/")
        print(f"       └── data/")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    # Check if path provided via command line
    if len(sys.argv) > 1:
        main(project_root=sys.argv[1])
    else:
        main()


# For notebook usage, call like this:
main(project_root='C:/Users/sadeghi.a/Desktop/farsi-faker')
