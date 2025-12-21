"""Core faker module for generating Persian/Farsi names.

This module provides the FarsiFaker class for generating authentic Persian/Iranian
names with gender specification and various configuration options for testing,
mock data generation, and development purposes.
"""

import random
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Literal, Tuple, Union

# Type aliases for better code clarity
GenderType = Literal['male', 'female']
GenderInput = Union[str, None]


class FarsiFaker:
    """High-performance faker for authentic Persian/Farsi names.
    
    This class provides methods to generate realistic Persian/Farsi names with
    support for gender specification, reproducible results, and various output formats.
    
    The class uses optimized pickle-based data storage for fast loading and includes
    10,000+ authentic Persian names sourced from real Iranian datasets.
    
    Attributes:
        _male_names (List[str]): List of male first names
        _female_names (List[str]): List of female first names
        _last_names (List[str]): List of family names
        _random (random.Random): Random number generator instance
    
    Example:
        >>> from farsi_faker import FarsiFaker
        >>> 
        >>> faker = FarsiFaker(seed=42)
        >>> person = faker.full_name('male')
        >>> print(person)
        {
            'name': 'علی احمدی',
            'first_name': 'علی',
            'last_name': 'احمدی',
            'gender': 'male'
        }
        >>> 
        >>> # Generate 10 female names
        >>> women = faker.generate_names(10, 'female')
        >>> 
        >>> # Generate balanced dataset
        >>> dataset = faker.generate_dataset(100, male_ratio=0.5)
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
        """Initialize the Farsi faker.
        
        Args:
            seed: Optional random seed for reproducible results. If None,
                  uses system randomness for non-deterministic generation.
        
        Raises:
            FileNotFoundError: If the names data file cannot be found.
            pickle.UnpicklingError: If the data file is corrupted.
        
        Example:
            >>> # Random generation
            >>> faker = FarsiFaker()
            >>> 
            >>> # Reproducible generation
            >>> faker = FarsiFaker(seed=42)
            >>> name1 = faker.full_name()
            >>> faker = FarsiFaker(seed=42)  # Reset with same seed
            >>> name2 = faker.full_name()
            >>> assert name1 == name2  # Same results
        """
        self._random = random.Random(seed)
        self._load_data()
    
    def _load_data(self) -> None:
        """Load names data from pickle file with class-level caching.
        
        Uses class-level caching to avoid redundant file I/O operations when
        creating multiple faker instances. The data is loaded once and shared
        across all instances for optimal memory usage and performance.
        
        Raises:
            FileNotFoundError: If names.pkl doesn't exist in the data directory.
            pickle.UnpicklingError: If the pickle file is corrupted or invalid.
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
            except Exception as e:
                raise pickle.UnpicklingError(
                    f"Failed to load names data: {e}\n"
                    "The data file may be corrupted. Try reinstalling the package:\n"
                    "pip install --force-reinstall farsi-faker"
                )
        
        # Use cached data
        self._male_names = FarsiFaker._data_cache['male_names']
        self._female_names = FarsiFaker._data_cache['female_names']
        self._last_names = FarsiFaker._data_cache['last_names']
    
    def _normalize_gender(self, gender: GenderInput) -> Optional[GenderType]:
        """Normalize gender input to standard format.
        
        Accepts various gender specifications in both Persian and English
        and normalizes them to 'male' or 'female'.
        
        Args:
            gender: Gender specification in various formats:
                   - English: 'male', 'female', 'm', 'f'
                   - Persian: 'مرد', 'زن', 'پسر', 'دختر', 'مذکر', 'مونث'
                   - None for random selection
        
        Returns:
            Normalized gender ('male' or 'female'), or None if input is None
        
        Raises:
            ValueError: If gender value is not recognized
        
        Example:
            >>> faker = FarsiFaker()
            >>> faker._normalize_gender('مرد')
            'male'
            >>> faker._normalize_gender('f')
            'female'
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
    
    def male_first_name(self) -> str:
        """Generate a random male first name.
        
        Returns:
            A randomly selected authentic male Persian name.
        
        Example:
            >>> faker = FarsiFaker()
            >>> name = faker.male_first_name()
            >>> print(name)
            محمد
        """
        return self._random.choice(self._male_names)
    
    def female_first_name(self) -> str:
        """Generate a random female first name.
        
        Returns:
            A randomly selected authentic female Persian name.
        
        Example:
            >>> faker = FarsiFaker()
            >>> name = faker.female_first_name()
            >>> print(name)
            فاطمه
        """
        return self._random.choice(self._female_names)
    
    def first_name(self, gender: GenderInput = None) -> Tuple[str, GenderType]:
        """Generate a first name with optional gender specification.
        
        Args:
            gender: Desired gender ('male', 'female', or Persian equivalents).
                   If None, randomly selects between male and female.
        
        Returns:
            Tuple of (name, gender) where gender is normalized to 'male' or 'female'.
        
        Raises:
            ValueError: If gender is invalid.
        
        Example:
            >>> faker = FarsiFaker()
            >>> name, gender = faker.first_name('male')
            >>> print(f"{name} ({gender})")
            علی (male)
            >>> 
            >>> # Random gender
            >>> name, gender = faker.first_name()
            >>> print(f"{name} ({gender})")
            مریم (female)
        """
        normalized_gender = self._normalize_gender(gender)
        
        if normalized_gender is None:
            normalized_gender = self._random.choice(['male', 'female'])
        
        if normalized_gender == 'male':
            return (self.male_first_name(), 'male')
        else:
            return (self.female_first_name(), 'female')
    
    def last_name(self) -> str:
        """Generate a random family name.
        
        Returns:
            A randomly selected authentic Persian family name.
        
        Example:
            >>> faker = FarsiFaker()
            >>> name = faker.last_name()
            >>> print(name)
            احمدی
        """
        return self._random.choice(self._last_names)
    
    def full_name(self, gender: GenderInput = None) -> Dict[str, str]:
        """Generate a complete person with full name and metadata.
        
        Args:
            gender: Desired gender ('male', 'female', or Persian equivalents).
                   If None, randomly selects gender.
        
        Returns:
            Dictionary containing:
                - name: Full name (first + last)
                - first_name: First name only
                - last_name: Family name only
                - gender: Normalized gender ('male' or 'female')
        
        Raises:
            ValueError: If gender is invalid.
        
        Example:
            >>> faker = FarsiFaker()
            >>> person = faker.full_name('female')
            >>> print(person)
            {
                'name': 'فاطمه محمدی',
                'first_name': 'فاطمه',
                'last_name': 'محمدی',
                'gender': 'female'
            }
            >>> 
            >>> # Direct access
            >>> print(person['name'])
            فاطمه محمدی
        """
        first, gender_result = self.first_name(gender)
        last = self.last_name()
        
        return {
            'name': f"{first} {last}",
            'first_name': first,
            'last_name': last,
            'gender': gender_result
        }
    
    def generate_names(
        self,
        count: int = 10,
        gender: GenderInput = None
    ) -> List[Dict[str, str]]:
        """Generate multiple full names.
        
        Args:
            count: Number of names to generate (must be positive).
            gender: Desired gender for all names. If None, randomly mixes genders.
        
        Returns:
            List of person dictionaries (see full_name() for structure).
        
        Raises:
            ValueError: If count is not positive or gender is invalid.
        
        Example:
            >>> faker = FarsiFaker()
            >>> 
            >>> # Generate 5 male names
            >>> men = faker.generate_names(5, 'male')
            >>> for person in men:
            ...     print(person['name'])
            علی احمدی
            محمد رضایی
            حسین کریمی
            رضا محمدی
            احمد حسینی
            >>> 
            >>> # Generate 3 random gender names
            >>> people = faker.generate_names(3)
        """
        if count <= 0:
            raise ValueError(f"Count must be positive, got: {count}")
        
        return [self.full_name(gender) for _ in range(count)]
    
    def generate_dataset(
    self,
    count: int = 100,
    male_ratio: float = 0.5
    ) -> List[Dict[str, str]]:
        """Generate a balanced dataset with specified gender ratio.
        
        Args:
            count: Total number of names to generate (must be positive).
            male_ratio: Ratio of male names (0.0 to 1.0). Default is 0.5 (balanced).
                Examples:
                - 0.5 = 50% male, 50% female (balanced)
                - 0.7 = 70% male, 30% female
                - 0.0 = 100% female
                - 1.0 = 100% male
        
        Returns:
            List of person dictionaries in random (shuffled) order.
        
        Raises:
            ValueError: If count is not positive or male_ratio is out of range.
        """
        if count <= 0:
            raise ValueError(f"Count must be positive, got: {count}")
        
        if not 0 <= male_ratio <= 1:
            raise ValueError(
                f"male_ratio must be between 0 and 1, got: {male_ratio}\n"
                "Examples: 0.5 (balanced), 0.7 (70% male), 1.0 (all male)"
            )
        
        male_count = int(count * male_ratio)
        female_count = count - male_count
        
        dataset = []
        
        # Only generate if count > 0
        if male_count > 0:
            dataset.extend(self.generate_names(male_count, 'male'))
        
        if female_count > 0:
            dataset.extend(self.generate_names(female_count, 'female'))
        
        # Shuffle to mix genders randomly
        self._random.shuffle(dataset)
        
        return dataset

    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the names database.
        
        Returns:
            Dictionary containing:
                - male_names_count: Number of unique male first names
                - female_names_count: Number of unique female first names
                - last_names_count: Number of unique family names
                - total_names: Sum of all unique names
                - possible_combinations: Total possible full name combinations
        
        Example:
            >>> faker = FarsiFaker()
            >>> stats = faker.get_stats()
            >>> print(f"Male names: {stats['male_names_count']:,}")
            Male names: 3,500
            >>> print(f"Possible combinations: {stats['possible_combinations']:,}")
            Possible combinations: 21,000,000
        """
        male_count = len(self._male_names)
        female_count = len(self._female_names)
        last_count = len(self._last_names)
        
        return {
            'male_names_count': male_count,
            'female_names_count': female_count,
            'last_names_count': last_count,
            'total_names': male_count + female_count + last_count,
            'possible_combinations': (male_count + female_count) * last_count
        }


# Convenience function for quick one-off name generation
def generate_fake_name(gender: GenderInput = None, seed: Optional[int] = None) -> Dict[str, str]:
    """Quick function to generate a single fake Persian name.
    
    This is a convenience function that creates a faker instance and generates
    one name. For generating multiple names, create a FarsiFaker instance
    directly for better performance.
    
    Args:
        gender: Desired gender ('male', 'female', or Persian equivalents).
        seed: Optional random seed for reproducibility.
    
    Returns:
        Person dictionary with full name and metadata.
    
    Example:
        >>> from farsi_faker import generate_fake_name
        >>> 
        >>> # Quick male name
        >>> person = generate_fake_name('male')
        >>> print(person['name'])
        علی احمدی
        >>> 
        >>> # Reproducible
        >>> person1 = generate_fake_name('female', seed=123)
        >>> person2 = generate_fake_name('female', seed=123)
        >>> assert person1 == person2
    """
    faker = FarsiFaker(seed=seed)
    return faker.full_name(gender)
