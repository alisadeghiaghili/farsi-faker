"""Core faker module for generating Persian/Farsi names.

This module provides the FarsiFaker class for generating authentic Persian/Iranian
names with gender specification and various configuration options for testing,
mock data generation, and development purposes.
"""

import random
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    import pandas as pd

# Type aliases for better code clarity
GenderType = Literal['male', 'female']
GenderInput = Union[str, None]


class FarsiFaker:
    """High-performance faker for authentic Persian/Farsi names.

    This class provides methods to generate realistic Persian/Farsi names with
    support for gender specification, reproducible results, and various output
    formats including plain Python lists and optional pandas DataFrames.

    The class uses optimized pickle-based data storage for fast loading and
    includes 10,000+ authentic Persian names sourced from real Iranian datasets.
    Name data is loaded once and cached at the class level so that creating
    multiple FarsiFaker instances does not cause redundant file I/O.

    Attributes:
        _male_names (List[str]): List of male first names loaded from the
            embedded pickle database.
        _female_names (List[str]): List of female first names loaded from the
            embedded pickle database.
        _last_names (List[str]): List of family names loaded from the embedded
            pickle database.
        _random (random.Random): Instance-level random number generator,
            seeded via the constructor ``seed`` parameter.

    Example:
        Basic usage::

            >>> from farsi_faker import FarsiFaker
            >>> faker = FarsiFaker(seed=42)

            >>> person = faker.full_name('male')
            >>> print(person)
            {'name': 'علی احمدی', 'first_name': 'علی', 'last_name': 'احمدی', 'gender': 'male'}

        Generate many names at once::

            >>> women = faker.generate_names(10, 'female')
            >>> print(len(women))
            10
            >>> print(women[0]['gender'])
            female

        pandas DataFrame output::

            >>> df = faker.generate_dataset(100, male_ratio=0.5, as_dataframe=True)
            >>> print(df.shape)
            (100, 4)
            >>> print(list(df.columns))
            ['name', 'first_name', 'last_name', 'gender']
    """

    # Class-level cache for data (shared across instances for memory efficiency)
    _data_cache: Optional[Dict[str, List[str]]] = None

    # Gender mapping for flexible input (supports Persian and English)
    _GENDER_MAP = {
        'male': 'male',
        'm': 'male',
        'مرد': 'male',
        'پسر': 'male',
        'مذکر': 'male',
        'female': 'female',
        'f': 'female',
        'زن': 'female',
        'دختر': 'female',
        'مونث': 'female',
    }

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the FarsiFaker instance.

        Creates an instance-level random number generator and loads the names
        database from the embedded pickle file.  The pickle data is cached at
        the class level after the first load, so subsequent instantiations are
        essentially free.

        Args:
            seed (int, optional): Random seed for reproducible results.
                Pass the same seed to two separate instances to guarantee
                identical output sequences.  If ``None`` (default), the
                instance uses system entropy and produces non-deterministic
                output.

        Raises:
            FileNotFoundError: If ``farsi_faker/data/names.pkl`` cannot be
                found, which typically means the package was not installed
                correctly.
            pickle.UnpicklingError: If the names pickle file is corrupted or
                was created with an incompatible pickle protocol.

        Example:
            Non-deterministic (default)::

                >>> faker = FarsiFaker()
                >>> faker.full_name()  # different every run
                {'name': '...', 'first_name': '...', 'last_name': '...', 'gender': '...'}

            Reproducible::

                >>> faker1 = FarsiFaker(seed=42)
                >>> faker2 = FarsiFaker(seed=42)
                >>> assert faker1.full_name() == faker2.full_name()
        """
        self._random = random.Random(seed)
        self._load_data()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load names data from pickle file with class-level caching.

        Called once per process.  Subsequent calls return immediately because
        ``_data_cache`` is already populated.

        Raises:
            FileNotFoundError: If ``data/names.pkl`` is missing.
            pickle.UnpicklingError: If the file exists but cannot be
                deserialized (e.g., corrupted or wrong protocol).
        """
        if FarsiFaker._data_cache is None:
            data_path = Path(__file__).parent / 'data' / 'names.pkl'

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Names data file not found: {data_path}\n"
                    "Please ensure the package is installed correctly.\n"
                    "Try reinstalling: pip install --force-reinstall farsi-faker"
                )

            try:
                with open(data_path, 'rb') as f:
                    FarsiFaker._data_cache = pickle.load(f)
            except Exception as exc:
                raise pickle.UnpicklingError(
                    f"Failed to load names data: {exc}\n"
                    "The data file may be corrupted. Try reinstalling:\n"
                    "pip install --force-reinstall farsi-faker"
                ) from exc

        self._male_names: List[str] = FarsiFaker._data_cache['male_names']
        self._female_names: List[str] = FarsiFaker._data_cache['female_names']
        self._last_names: List[str] = FarsiFaker._data_cache['last_names']

    def _normalize_gender(self, gender: GenderInput) -> Optional[GenderType]:
        """Normalize a raw gender string to ``'male'`` or ``'female'``.

        Accepts all supported Persian and English gender tokens and maps them
        to the canonical two-value enum used internally.  This is an internal
        helper; callers should use the public methods instead.

        Supported tokens:

        +------------------+----------+
        | Input token      | Result   |
        +==================+==========+
        | ``'male'``, ``'m'``         | ``'male'``   |
        +------------------+----------+
        | ``'مرد'``, ``'پسر'``,       | ``'male'``   |
        | ``'مذکر'``                  |              |
        +------------------+----------+
        | ``'female'``, ``'f'``       | ``'female'`` |
        +------------------+----------+
        | ``'زن'``, ``'دختر'``,       | ``'female'`` |
        | ``'مونث'``                  |              |
        +------------------+----------+
        | ``None``         | ``None`` |
        +------------------+----------+

        Args:
            gender (str or None): Raw gender token.  Leading/trailing
                whitespace and casing are normalised before the lookup.

        Returns:
            Optional[GenderType]: ``'male'``, ``'female'``, or ``None`` when
            the input is ``None``.

        Raises:
            ValueError: If the token is not ``None`` and is not found in the
                mapping table.  The error message lists all valid values.
        """
        if gender is None:
            return None

        gender_lower = str(gender).lower().strip()
        normalized = self._GENDER_MAP.get(gender_lower)

        if normalized is None:
            valid_values = ', '.join(f"'{v}'" for v in sorted(set(self._GENDER_MAP.keys())))
            raise ValueError(
                f"Invalid gender: '{gender}'\n"
                f"Valid values: {valid_values}"
            )

        return normalized

    # ------------------------------------------------------------------
    # Public API — single-item generators
    # ------------------------------------------------------------------

    def male_first_name(self) -> str:
        """Return a random male first name.

        Returns:
            str: A randomly selected authentic male Persian first name.

        Example::

            >>> faker = FarsiFaker(seed=0)
            >>> faker.male_first_name()
            'محمد'
        """
        return self._random.choice(self._male_names)

    def female_first_name(self) -> str:
        """Return a random female first name.

        Returns:
            str: A randomly selected authentic female Persian first name.

        Example::

            >>> faker = FarsiFaker(seed=0)
            >>> faker.female_first_name()
            'فاطمه'
        """
        return self._random.choice(self._female_names)

    def first_name(self, gender: GenderInput = None) -> Tuple[str, GenderType]:
        """Return a first name together with its normalised gender.

        Args:
            gender (str, optional): Desired gender.  Any value accepted by
                :meth:`_normalize_gender` is valid (English or Persian tokens).
                When ``None`` (default) the gender is chosen at random with
                equal probability.

        Returns:
            Tuple[str, GenderType]: A 2-tuple ``(name, gender)`` where
            *gender* is always ``'male'`` or ``'female'``.

        Raises:
            ValueError: If *gender* is not a recognised token.

        Example::

            >>> faker = FarsiFaker(seed=1)
            >>> name, g = faker.first_name('male')
            >>> print(name, g)
            علی male

            >>> name, g = faker.first_name('زن')
            >>> print(g)
            female

            >>> name, g = faker.first_name()  # random gender
            >>> g in ('male', 'female')
            True
        """
        normalized_gender = self._normalize_gender(gender)

        if normalized_gender is None:
            normalized_gender = self._random.choice(['male', 'female'])

        if normalized_gender == 'male':
            return (self.male_first_name(), 'male')
        else:
            return (self.female_first_name(), 'female')

    def last_name(self) -> str:
        """Return a random Persian family name.

        Returns:
            str: A randomly selected authentic Persian family name.

        Example::

            >>> faker = FarsiFaker(seed=0)
            >>> faker.last_name()
            'احمدی'
        """
        return self._random.choice(self._last_names)

    def full_name(self, gender: GenderInput = None) -> Dict[str, str]:
        """Return a complete person record with full name and metadata.

        Combines :meth:`first_name` and :meth:`last_name` into a single dict
        that is ready to use as a test fixture or seed record.

        Args:
            gender (str, optional): Desired gender.  Any value accepted by
                :meth:`_normalize_gender` is valid.  When ``None`` (default)
                the gender is chosen at random.

        Returns:
            Dict[str, str]: A dictionary with exactly four keys:

            * ``'name'`` — full name (``first_name + ' ' + last_name``)
            * ``'first_name'`` — first name only
            * ``'last_name'`` — family name only
            * ``'gender'`` — ``'male'`` or ``'female'``

        Raises:
            ValueError: If *gender* is not a recognised token.

        Example::

            >>> faker = FarsiFaker(seed=7)
            >>> person = faker.full_name('female')
            >>> person['gender']
            'female'
            >>> person['name'] == person['first_name'] + ' ' + person['last_name']
            True

            >>> # Keys are always present and non-empty
            >>> all(person[k] for k in ('name', 'first_name', 'last_name', 'gender'))
            True
        """
        first, gender_result = self.first_name(gender)
        last = self.last_name()

        return {
            'name': f"{first} {last}",
            'first_name': first,
            'last_name': last,
            'gender': gender_result,
        }

    # ------------------------------------------------------------------
    # Public API — bulk generators
    # ------------------------------------------------------------------

    def generate_names(
        self,
        count: int = 10,
        gender: GenderInput = None,
        as_dataframe: bool = False,
    ) -> Union[List[Dict[str, str]], 'pd.DataFrame']:
        """Generate multiple full-name records.

        Args:
            count (int, optional): Number of records to generate.  Must be a
                positive integer.  Defaults to ``10``.
            gender (str, optional): Gender applied to *all* records.  When
                ``None`` (default) each record's gender is chosen independently
                at random.
            as_dataframe (bool, optional): When ``True``, return a
                ``pandas.DataFrame`` instead of a plain list.  Requires pandas
                to be installed (``pip install pandas``).  Defaults to
                ``False``.

        Returns:
            List[Dict[str, str]] | pandas.DataFrame:
                * **List** (default) — each element is a dict produced by
                  :meth:`full_name`::

                      [{'name': 'علی احمدی', 'first_name': 'علی',
                        'last_name': 'احمدی', 'gender': 'male'}, ...]

                * **DataFrame** (``as_dataframe=True``) — shape ``(count, 4)``,
                  columns ``['name', 'first_name', 'last_name', 'gender']``,
                  all dtype ``object``.

        Raises:
            ValueError: If *count* ≤ 0 or *gender* is unrecognised.
            ImportError: If ``as_dataframe=True`` but pandas is not installed.

        Example:
            List output (default)::

                >>> faker = FarsiFaker(seed=42)
                >>> men = faker.generate_names(3, 'male')
                >>> len(men)
                3
                >>> all(p['gender'] == 'male' for p in men)
                True
                >>> men[0].keys()
                dict_keys(['name', 'first_name', 'last_name', 'gender'])

            DataFrame output::

                >>> df = faker.generate_names(50, as_dataframe=True)
                >>> df.shape
                (50, 4)
                >>> list(df.columns)
                ['name', 'first_name', 'last_name', 'gender']
                >>> df.isnull().any().any()
                False
                >>> (df['name'] == df['first_name'] + ' ' + df['last_name']).all()
                True
        """
        if count <= 0:
            raise ValueError(f"count must be a positive integer, got: {count}")

        records = [self.full_name(gender) for _ in range(count)]

        if as_dataframe:
            try:
                import pandas as _pd
            except ImportError as exc:
                raise ImportError(
                    "pandas is required when as_dataframe=True.\n"
                    "Install it with: pip install pandas"
                ) from exc
            return _pd.DataFrame(
                records, columns=['name', 'first_name', 'last_name', 'gender']
            )

        return records

    def generate_dataset(
        self,
        count: int = 100,
        male_ratio: float = 0.5,
        as_dataframe: bool = False,
    ) -> Union[List[Dict[str, str]], 'pd.DataFrame']:
        """Generate a gender-balanced dataset with a configurable male/female ratio.

        Internally calls :meth:`generate_names` for each gender bucket, then
        shuffles the combined list so that gender order is random.

        Args:
            count (int, optional): Total number of records.  Must be a positive
                integer.  Defaults to ``100``.
            male_ratio (float, optional): Fraction of records that should be
                male, in the closed interval ``[0.0, 1.0]``.  Defaults to
                ``0.5`` (balanced).  Examples:

                * ``0.5``  → 50 % male, 50 % female
                * ``0.7``  → 70 % male, 30 % female
                * ``0.0``  → all female
                * ``1.0``  → all male

                The male count is computed as ``int(count * male_ratio)``;
                the remainder goes to female, so floating-point rounding is
                absorbed by the female bucket.
            as_dataframe (bool, optional): When ``True``, return a
                ``pandas.DataFrame`` instead of a plain list.  Requires pandas
                to be installed (``pip install pandas``).  Defaults to
                ``False``.

        Returns:
            List[Dict[str, str]] | pandas.DataFrame:
                * **List** (default) — shuffled list of person dicts (see
                  :meth:`full_name` for the dict structure).
                * **DataFrame** (``as_dataframe=True``) — shape
                  ``(count, 4)``, columns
                  ``['name', 'first_name', 'last_name', 'gender']``,
                  all dtype ``object``.

        Raises:
            ValueError: If *count* ≤ 0, or *male_ratio* is outside
                ``[0.0, 1.0]``.  The error message shows the computed male /
                female counts and the total so the problem is immediately
                obvious.
            ImportError: If ``as_dataframe=True`` but pandas is not installed.

        Example:
            List output (default)::

                >>> faker = FarsiFaker(seed=42)
                >>> dataset = faker.generate_dataset(10, male_ratio=0.6)
                >>> len(dataset)
                10
                >>> sum(1 for p in dataset if p['gender'] == 'male')
                6

            DataFrame output::

                >>> df = faker.generate_dataset(100, male_ratio=0.5, as_dataframe=True)
                >>> df.shape
                (100, 4)
                >>> df['gender'].value_counts().to_dict()
                {'male': 50, 'female': 50}
                >>> list(df.columns)
                ['name', 'first_name', 'last_name', 'gender']
                >>> df.isnull().any().any()
                False

            Edge cases::

                >>> all_female = faker.generate_dataset(5, male_ratio=0.0)
                >>> all(p['gender'] == 'female' for p in all_female)
                True

                >>> all_male = faker.generate_dataset(5, male_ratio=1.0)
                >>> all(p['gender'] == 'male' for p in all_male)
                True
        """
        if count <= 0:
            raise ValueError(f"count must be a positive integer, got: {count}")

        if not 0.0 <= male_ratio <= 1.0:
            raise ValueError(
                f"male_ratio must be between 0.0 and 1.0, got: {male_ratio}\n"
                f"This would generate {count * male_ratio:.1f} male and "
                f"{count * (1 - male_ratio):.1f} female names out of {count} total.\n"
                "Examples: 0.5 (balanced), 0.7 (70 % male), 1.0 (all male)"
            )

        male_count = int(count * male_ratio)
        female_count = count - male_count

        dataset: List[Dict[str, str]] = []
        if male_count > 0:
            dataset.extend(self.generate_names(male_count, 'male'))
        if female_count > 0:
            dataset.extend(self.generate_names(female_count, 'female'))

        self._random.shuffle(dataset)

        if as_dataframe:
            try:
                import pandas as _pd
            except ImportError as exc:
                raise ImportError(
                    "pandas is required when as_dataframe=True.\n"
                    "Install it with: pip install pandas"
                ) from exc
            return _pd.DataFrame(
                dataset, columns=['name', 'first_name', 'last_name', 'gender']
            )

        return dataset

    # ------------------------------------------------------------------
    # Public API — statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return statistics about the embedded names database.

        All counts reflect the number of *unique* entries in the pickle
        database.  ``possible_combinations`` is the cartesian product of
        first-name candidates (male + female) and family names, i.e. the
        theoretical upper bound on distinct full names.

        Returns:
            Dict[str, int]: A dictionary with exactly five keys:

            * ``'male_names_count'`` — unique male first names
            * ``'female_names_count'`` — unique female first names
            * ``'last_names_count'`` — unique family names
            * ``'total_names'`` — sum of the three counts above
            * ``'possible_combinations'`` —
              ``(male_names_count + female_names_count) * last_names_count``

        Example::

            >>> faker = FarsiFaker()
            >>> stats = faker.get_stats()
            >>> stats['male_names_count'] > 0
            True
            >>> stats['female_names_count'] > 0
            True
            >>> stats['possible_combinations'] == \\
            ...     (stats['male_names_count'] + stats['female_names_count']) \\
            ...     * stats['last_names_count']
            True
        """
        male_count = len(self._male_names)
        female_count = len(self._female_names)
        last_count = len(self._last_names)

        return {
            'male_names_count': male_count,
            'female_names_count': female_count,
            'last_names_count': last_count,
            'total_names': male_count + female_count + last_count,
            'possible_combinations': (male_count + female_count) * last_count,
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def generate_fake_name(
    gender: GenderInput = None,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    """Generate a single fake Persian name without managing a FarsiFaker instance.

    This is a convenience wrapper around :class:`FarsiFaker` for one-off
    name generation.  When generating many names in a loop, prefer creating
    a :class:`FarsiFaker` instance directly — it avoids the per-call
    constructor overhead.

    Args:
        gender (str, optional): Desired gender.  Accepts all tokens
            recognised by :meth:`FarsiFaker._normalize_gender`
            (English and Persian).  When ``None`` (default) the gender is
            chosen at random.
        seed (int, optional): Random seed for reproducible output.
            Two calls with the same *seed* and *gender* will return the
            same dict.  Defaults to ``None`` (non-deterministic).

    Returns:
        Dict[str, str]: Person record — identical structure to
        :meth:`FarsiFaker.full_name`:

        * ``'name'`` — full name
        * ``'first_name'`` — first name
        * ``'last_name'`` — family name
        * ``'gender'`` — ``'male'`` or ``'female'``

    Raises:
        ValueError: If *gender* is not a recognised token.

    Example:
        Quick male name::

            >>> from farsi_faker import generate_fake_name
            >>> person = generate_fake_name('male')
            >>> person['gender']
            'male'
            >>> all(k in person for k in ('name', 'first_name', 'last_name', 'gender'))
            True

        Reproducible::

            >>> p1 = generate_fake_name('female', seed=99)
            >>> p2 = generate_fake_name('female', seed=99)
            >>> p1 == p2
            True

        One-off vs. instance (performance note)::

            >>> # Prefer this for bulk generation
            >>> faker = FarsiFaker()
            >>> names = [faker.full_name() for _ in range(1000)]
    """
    return FarsiFaker(seed=seed).full_name(gender)
