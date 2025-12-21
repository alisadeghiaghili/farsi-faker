"""Comprehensive tests for FarsiFaker."""

import pytest
from farsi_faker import FarsiFaker, generate_fake_name


class TestFarsiFaker:
    """Test suite for FarsiFaker class."""
    
    @pytest.fixture
    def faker(self):
        """Create a faker instance with fixed seed for reproducible tests."""
        return FarsiFaker(seed=42)
    
    def test_initialization(self):
        """Test faker initialization."""
        faker = FarsiFaker()
        assert faker is not None
        assert hasattr(faker, '_male_names')
        assert hasattr(faker, '_female_names')
        assert hasattr(faker, '_last_names')
    
    def test_initialization_with_seed(self):
        """Test faker initialization with seed."""
        faker = FarsiFaker(seed=42)
        assert faker is not None
        assert faker._random is not None
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        faker1 = FarsiFaker(seed=123)
        faker2 = FarsiFaker(seed=123)
        
        name1 = faker1.full_name('male')
        name2 = faker2.full_name('male')
        
        assert name1 == name2
    
    def test_male_first_name(self, faker):
        """Test male first name generation."""
        name = faker.male_first_name()
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_female_first_name(self, faker):
        """Test female first name generation."""
        name = faker.female_first_name()
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_last_name(self, faker):
        """Test last name generation."""
        name = faker.last_name()
        assert isinstance(name, str)
        assert len(name) > 0
    
    @pytest.mark.parametrize("gender_input,expected", [
        ('male', 'male'),
        ('m', 'male'),
        ('مرد', 'male'),
        ('پسر', 'male'),
        ('مذکر', 'male'),
        ('female', 'female'),
        ('f', 'female'),
        ('زن', 'female'),
        ('دختر', 'female'),
        ('مونث', 'female'),
    ])
    def test_first_name_gender_variants(self, faker, gender_input, expected):
        """Test that all gender input variants work correctly."""
        name, gender = faker.first_name(gender_input)
        assert isinstance(name, str)
        assert len(name) > 0
        assert gender == expected
    
    def test_first_name_random_gender(self, faker):
        """Test random gender selection."""
        name, gender = faker.first_name()
        assert isinstance(name, str)
        assert len(name) > 0
        assert gender in ['male', 'female']
    
    def test_first_name_invalid_gender(self, faker):
        """Test that invalid gender raises ValueError."""
        with pytest.raises(ValueError, match="Invalid gender"):
            faker.first_name('invalid')
    
    def test_first_name_case_insensitive(self, faker):
        """Test that gender input is case-insensitive."""
        name1, gender1 = faker.first_name('MALE')
        name2, gender2 = faker.first_name('Male')
        name3, gender3 = faker.first_name('male')
        
        assert gender1 == gender2 == gender3 == 'male'
    
    def test_full_name_structure(self, faker):
        """Test full name dictionary structure."""
        person = faker.full_name('male')
        
        assert isinstance(person, dict)
        assert 'name' in person
        assert 'first_name' in person
        assert 'last_name' in person
        assert 'gender' in person
        
        assert person['gender'] == 'male'
        assert person['name'] == f"{person['first_name']} {person['last_name']}"
        assert len(person['first_name']) > 0
        assert len(person['last_name']) > 0
    
    def test_full_name_contains_space(self, faker):
        """Test that full name contains exactly one space."""
        person = faker.full_name()
        assert person['name'].count(' ') == 1
    
    def test_generate_names_count(self, faker):
        """Test generating multiple names."""
        count = 50
        names = faker.generate_names(count, 'male')
        
        assert len(names) == count
        assert all(isinstance(n, dict) for n in names)
        assert all(n['gender'] == 'male' for n in names)
    
    def test_generate_names_invalid_count(self, faker):
        """Test that invalid count raises ValueError."""
        with pytest.raises(ValueError, match="Count must be positive"):
            faker.generate_names(-5)
        
        with pytest.raises(ValueError, match="Count must be positive"):
            faker.generate_names(0)
    
    def test_generate_names_mixed_gender(self, faker):
        """Test generating names with mixed genders."""
        names = faker.generate_names(20)  # No gender specified
        
        assert len(names) == 20
        genders = set(n['gender'] for n in names)
        # Should have both genders (with high probability)
        assert len(genders) >= 1  # At least one gender
    
    def test_generate_dataset_ratio(self, faker):
        """Test dataset generation with specific ratio."""
        total = 100
        male_ratio = 0.7
        dataset = faker.generate_dataset(total, male_ratio)
        
        assert len(dataset) == total
        
        males = sum(1 for p in dataset if p['gender'] == 'male')
        females = total - males
        
        # Allow small deviation due to rounding
        expected_males = int(total * male_ratio)
        assert abs(males - expected_males) <= 1
    
    def test_generate_dataset_edge_cases(self, faker):
        """Test dataset with edge case ratios."""
        # All male
        dataset_all_male = faker.generate_dataset(50, male_ratio=1.0)
        assert all(p['gender'] == 'male' for p in dataset_all_male)
        
        # All female
        dataset_all_female = faker.generate_dataset(50, male_ratio=0.0)
        assert all(p['gender'] == 'female' for p in dataset_all_female)
        
        # Balanced
        dataset_balanced = faker.generate_dataset(100, male_ratio=0.5)
        males = sum(1 for p in dataset_balanced if p['gender'] == 'male')
        assert 45 <= males <= 55  # Allow some deviation
    
    def test_generate_dataset_invalid_ratio(self, faker):
        """Test that invalid ratio raises ValueError."""
        with pytest.raises(ValueError, match="male_ratio must be between 0 and 1"):
            faker.generate_dataset(100, male_ratio=1.5)
        
        with pytest.raises(ValueError, match="male_ratio must be between 0 and 1"):
            faker.generate_dataset(100, male_ratio=-0.1)
    
    def test_generate_dataset_invalid_count(self, faker):
        """Test that invalid count raises ValueError."""
        with pytest.raises(ValueError, match="Count must be positive"):
            faker.generate_dataset(-10)
        
        with pytest.raises(ValueError, match="Count must be positive"):
            faker.generate_dataset(0)
    
    def test_generate_dataset_is_shuffled(self):
        """Test that dataset is properly shuffled."""
        faker = FarsiFaker(seed=42)
        dataset = faker.generate_dataset(100, male_ratio=0.5)
        
        # Check that genders are not all grouped together
        first_half_males = sum(1 for p in dataset[:50] if p['gender'] == 'male')
        second_half_males = sum(1 for p in dataset[50:] if p['gender'] == 'male')
        
        # If properly shuffled, both halves should have some males
        # (not a perfect test but good enough)
        assert first_half_males > 0
        assert second_half_males > 0
    
    def test_get_stats(self, faker):
        """Test statistics retrieval."""
        stats = faker.get_stats()
        
        assert 'male_names_count' in stats
        assert 'female_names_count' in stats
        assert 'last_names_count' in stats
        assert 'total_names' in stats
        assert 'possible_combinations' in stats
        
        assert stats['male_names_count'] > 0
        assert stats['female_names_count'] > 0
        assert stats['last_names_count'] > 0
        
        expected_total = (
            stats['male_names_count'] +
            stats['female_names_count'] +
            stats['last_names_count']
        )
        assert stats['total_names'] == expected_total
        
        expected_combinations = (
            (stats['male_names_count'] + stats['female_names_count']) *
            stats['last_names_count']
        )
        assert stats['possible_combinations'] == expected_combinations
    
    def test_get_stats_values_are_positive(self, faker):
        """Test that all stats values are positive integers."""
        stats = faker.get_stats()
        
        for key, value in stats.items():
            assert isinstance(value, int)
            assert value > 0
    
    def test_data_caching(self):
        """Test that data is cached across instances."""
        faker1 = FarsiFaker()
        faker2 = FarsiFaker()
        
        # Both should use same cached data (memory efficiency)
        assert faker1._male_names is faker2._male_names
        assert faker1._female_names is faker2._female_names
        assert faker1._last_names is faker2._last_names
    
    def test_data_immutability(self, faker):
        """Test that name lists are not accidentally modified."""
        original_male_count = len(faker._male_names)
        original_female_count = len(faker._female_names)
        original_last_count = len(faker._last_names)
        
        # Generate some names
        faker.generate_dataset(100)
        
        # Counts should remain the same
        assert len(faker._male_names) == original_male_count
        assert len(faker._female_names) == original_female_count
        assert len(faker._last_names) == original_last_count
    
    def test_convenience_function(self):
        """Test the convenience function."""
        person = generate_fake_name('male', seed=42)
        
        assert isinstance(person, dict)
        assert person['gender'] == 'male'
        assert 'name' in person
        assert 'first_name' in person
        assert 'last_name' in person
    
    def test_convenience_function_reproducibility(self):
        """Test convenience function reproducibility."""
        person1 = generate_fake_name('female', seed=999)
        person2 = generate_fake_name('female', seed=999)
        
        assert person1 == person2
    
    def test_convenience_function_without_seed(self):
        """Test convenience function without seed."""
        person = generate_fake_name('male')
        
        assert isinstance(person, dict)
        assert person['gender'] == 'male'


class TestIntegration:
    """Integration tests for real-world usage scenarios."""
    
    def test_csv_export_scenario(self):
        """Test generating data for CSV export."""
        faker = FarsiFaker(seed=100)
        dataset = faker.generate_dataset(50, male_ratio=0.6)
        
        # Verify we can iterate and access all fields
        for person in dataset:
            assert person['name']
            assert person['first_name']
            assert person['last_name']
            assert person['gender'] in ['male', 'female']
    
    def test_large_dataset_generation(self):
        """Test generating large datasets efficiently."""
        faker = FarsiFaker()
        dataset = faker.generate_dataset(1000, male_ratio=0.5)
        
        assert len(dataset) == 1000
        males = sum(1 for p in dataset if p['gender'] == 'male')
        # Allow 5% deviation for large datasets
        assert 450 <= males <= 550
    
    def test_multiple_faker_instances(self):
        """Test using multiple faker instances simultaneously."""
        faker1 = FarsiFaker(seed=1)
        faker2 = FarsiFaker(seed=2)
        faker3 = FarsiFaker()  # Random
        
        name1 = faker1.full_name()
        name2 = faker2.full_name()
        name3 = faker3.full_name()
        
        # Should work without interference
        assert name1 != name2  # Different seeds
        assert all(isinstance(n, dict) for n in [name1, name2, name3])
    
    def test_realistic_usage_pattern(self):
        """Test a realistic usage pattern."""
        # Create faker
        faker = FarsiFaker(seed=42)
        
        # Generate individuals
        person1 = faker.full_name('male')
        person2 = faker.full_name('female')
        
        # Generate batch
        batch = faker.generate_names(10, 'male')
        
        # Generate dataset
        dataset = faker.generate_dataset(50, male_ratio=0.6)
        
        # All should work together
        assert person1['gender'] == 'male'
        assert person2['gender'] == 'female'
        assert len(batch) == 10
        assert len(dataset) == 50
    
    def test_unicode_handling(self):
        """Test that Persian/Unicode characters are handled correctly."""
        faker = FarsiFaker()
        person = faker.full_name()
        
        # Name should be valid Unicode string
        assert isinstance(person['name'], str)
        assert isinstance(person['first_name'], str)
        assert isinstance(person['last_name'], str)
        
        # Should be able to encode/decode
        name_bytes = person['name'].encode('utf-8')
        assert name_bytes.decode('utf-8') == person['name']


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_gender_types(self):
        """Test various invalid gender inputs."""
        faker = FarsiFaker()
        
        invalid_genders = ['invalid', 'xyz', '123', 'man', 'woman']
        
        for invalid_gender in invalid_genders:
            with pytest.raises(ValueError):
                faker.first_name(invalid_gender)
    
    def test_boundary_conditions(self):
        """Test boundary conditions for count and ratio."""
        faker = FarsiFaker()
        
        # Minimum valid values
        assert len(faker.generate_names(1)) == 1
        assert len(faker.generate_dataset(1, male_ratio=0.0)) == 1
        assert len(faker.generate_dataset(1, male_ratio=1.0)) == 1
        
        # Edge ratios
        dataset_min = faker.generate_dataset(10, male_ratio=0.0)
        assert all(p['gender'] == 'female' for p in dataset_min)
        
        dataset_max = faker.generate_dataset(10, male_ratio=1.0)
        assert all(p['gender'] == 'male' for p in dataset_max)
    
    def test_zero_and_negative_inputs(self):
        """Test that zero and negative inputs are rejected."""
        faker = FarsiFaker()
        
        with pytest.raises(ValueError):
            faker.generate_names(0)
        
        with pytest.raises(ValueError):
            faker.generate_names(-1)
        
        with pytest.raises(ValueError):
            faker.generate_dataset(0)
        
        with pytest.raises(ValueError):
            faker.generate_dataset(-1)


class TestPerformance:
    """Performance-related tests."""
    
    def test_large_batch_performance(self):
        """Test that large batch generation completes in reasonable time."""
        import time
        
        faker = FarsiFaker()
        start = time.time()
        
        # Generate 10,000 names
        dataset = faker.generate_dataset(10000, male_ratio=0.5)
        
        elapsed = time.time() - start
        
        assert len(dataset) == 10000
        # Should complete in less than 5 seconds
        assert elapsed < 5.0
    
    def test_memory_efficiency(self):
        """Test that multiple instances don't duplicate data."""
        import sys
        
        # Create first instance
        faker1 = FarsiFaker()
        
        # Get memory size of data
        size1 = sys.getsizeof(faker1._male_names)
        
        # Create second instance
        faker2 = FarsiFaker()
        
        # Both should reference same data (cached)
        assert faker1._male_names is faker2._male_names
