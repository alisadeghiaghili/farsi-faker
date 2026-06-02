# Farsi Faker | فارسی فیکر

<div align="center">

<a href="https://pypi.org/project/farsi-faker/">
    <img src="https://badge.fury.io/py/farsi-faker.svg" alt="PyPI version">
</a>
<a href="https://pypi.org/project/farsi-faker/">
    <img src="https://img.shields.io/pypi/pyversions/farsi-faker.svg" alt="Python Support">
</a>
<a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
</a>
<a href="https://pepy.tech/project/farsi-faker">
    <img src="https://static.pepy.tech/personalized-badge/farsi-faker?period=total&units=international_system&left_color=black&right_color=green&left_text=downloads" alt="Downloads">
</a>
<a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
</a>

</div>

<div align="center">

**Generate realistic fake Persian/Farsi names for testing and development**

تولید اسم‌های فارسی فیک واقع‌گرایانه برای تست و توسعه

</div>

<div align="center">

[🌐 Website](https://alisadeghiaghili.github.io/farsi-faker/) • 
[📦 Installation](#-installation) • 
[🚀 Quick Start](#-quick-start) • 
[📖 Documentation](#-documentation) • 
[🎨 Examples](#-examples) • 
[🤝 Contributing](#-contributing)

</div>

---

## ✨ Features

- **🎯 10,000+ Authentic Names** - Real Persian names from Iranian datasets
- **👥 Gender-Specific** - Separate male and female name generation
- **⚡ High Performance** - Optimized pickle-based data storage
- **🔄 Reproducible** - Seed support for consistent results
- **🚀 Zero Dependencies** - No external packages required for production
- **🔒 Thread-Safe** - Safe for concurrent use
- **📝 Fully Typed** - Complete type hints for better IDE support
- **✅ Well Tested** - Comprehensive test coverage
- **🌍 Unicode Support** - Full Persian/Farsi character support
- **🐌 pandas Integration** - Optional DataFrame output for data science workflows

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install farsi-faker
```

### With pandas support (for DataFrame output)

```bash
pip install farsi-faker pandas
```

### From Source

```bash
git clone https://github.com/alisadeghiaghili/farsi-faker.git
cd farsi-faker
pip install -e .
```

### Requirements

- **Python 3.7+**
- **No external dependencies** for production use
- Optional: `pandas` for DataFrame output (`as_dataframe=True`)

---

## 🚀 Quick Start

### Basic Usage

```python
from farsi_faker import FarsiFaker

# Create faker instance
faker = FarsiFaker()

# Generate a random person
person = faker.full_name()
print(person)
# {'name': 'علی صادقی عقیلی', 'first_name': 'علی', 'last_name': 'صادقی عقیلی', 'gender': 'male'}

# Generate male name
male = faker.full_name('male')
print(male['name'])  # علی صادقی عقیلی

# Generate female name
female = faker.full_name('female')
print(female['name'])  # سپیده جلیلی
```

### Generate Multiple Names

```python
# Generate 10 random names (as list)
people = faker.generate_names(10)

# Generate 50 male names
men = faker.generate_names(50, 'male')

# Generate 30 female names as pandas DataFrame
import pandas as pd
women_df = faker.generate_names(30, 'female', as_dataframe=True)
print(women_df.head())
#          name first_name last_name  gender
# 0  فاطمه احمدی     فاطمه    احمدی  female
# 1  زینب رضایی      زینب    رضایی  female
```

### Generate Balanced Dataset

```python
# Generate 100 people with 60% male ratio (as list)
dataset = faker.generate_dataset(100, male_ratio=0.6)

# Generate as pandas DataFrame — ideal for data science workflows
df = faker.generate_dataset(500, male_ratio=0.5, as_dataframe=True)
print(df.shape)                    # (500, 4)
print(df['gender'].value_counts())
# male      250
# female    250
print(df.dtypes)
# name          object
# first_name    object
# last_name     object
# gender        object
```

### Reproducible Results

```python
# Use seed for reproducible results
faker1 = FarsiFaker(seed=42)
faker2 = FarsiFaker(seed=42)

name1 = faker1.full_name()
name2 = faker2.full_name()

assert name1 == name2  # True - same results!
```

### Quick One-Off Generation

```python
from farsi_faker import generate_fake_name

# Quick generation without creating instance
person = generate_fake_name('male')
print(person['name'])  # علی صادقی عقیلی
```

---

## 📖 Documentation

### Class: `FarsiFaker`

Main class for generating Persian names.

#### Constructor

```python
FarsiFaker(seed: Optional[int] = None)
```

**Parameters:**
- `seed` (int, optional): Random seed for reproducible results

**Example:**
```python
faker = FarsiFaker()  # Random generation
faker = FarsiFaker(seed=42)  # Reproducible generation
```

---

#### Methods

### `male_first_name() -> str`

Generate a random male first name.

**Returns:** Male Persian name as string

**Example:**
```python
name = faker.male_first_name()
# 'محمد'
```

---

### `female_first_name() -> str`

Generate a random female first name.

**Returns:** Female Persian name as string

**Example:**
```python
name = faker.female_first_name()
# 'فاطمه'
```

---

### `first_name(gender=None) -> Tuple[str, str]`

Generate a first name with optional gender specification.

**Parameters:**
- `gender` (str, optional): Gender ('male', 'female', 'm', 'f', 'مرد', 'زن', etc.)

**Returns:** Tuple of (name, normalized_gender)

**Example:**
```python
name, gender = faker.first_name('male')
# ('علی', 'male')

name, gender = faker.first_name()  # Random
# ('مریم', 'female')
```

---

### `last_name() -> str`

Generate a random family name.

**Returns:** Persian family name as string

**Example:**
```python
name = faker.last_name()
# 'احمدی'
```

---

### `full_name(gender=None) -> Dict[str, str]`

Generate a complete person with full name and metadata.

**Parameters:**
- `gender` (str, optional): Desired gender

**Returns:** Dictionary with keys:
- `name`: Full name
- `first_name`: First name only
- `last_name`: Family name only
- `gender`: Normalized gender ('male' or 'female')

**Example:**
```python
person = faker.full_name('female')
# {
#     'name': 'سپیده جلیلی',
#     'first_name': 'سپیده',
#     'last_name': 'جلیلی',
#     'gender': 'female'
# }
```

---

### `generate_names(count=10, gender=None, as_dataframe=False) -> List | DataFrame`

Generate multiple full names.

**Parameters:**
- `count` (int): Number of names to generate
- `gender` (str, optional): Gender for all names
- `as_dataframe` (bool): If `True`, returns a `pandas.DataFrame`. Default: `False`.

**Returns:** List of person dicts, or `pandas.DataFrame` when `as_dataframe=True`.
DataFrame columns: `name`, `first_name`, `last_name`, `gender`.

**Raises:** `ImportError` if `as_dataframe=True` and pandas is not installed.

**Example:**
```python
# List (default)
people = faker.generate_names(5, 'male')

# pandas DataFrame
df = faker.generate_names(100, as_dataframe=True)
print(df.shape)   # (100, 4)
print(df.dtypes)  # all object
```

---

### `generate_dataset(count=100, male_ratio=0.5, as_dataframe=False) -> List | DataFrame`

Generate a balanced dataset with specified gender ratio.

**Parameters:**
- `count` (int): Total number of names
- `male_ratio` (float): Ratio of male names (0.0 to 1.0)
- `as_dataframe` (bool): If `True`, returns a `pandas.DataFrame`. Default: `False`.

**Returns:** Shuffled list of person dicts, or `pandas.DataFrame` when `as_dataframe=True`.

**Raises:**
- `ValueError` if count is not positive or male_ratio is outside [0.0, 1.0]
- `ImportError` if `as_dataframe=True` and pandas is not installed

**Example:**
```python
# List (default)
dataset = faker.generate_dataset(100, male_ratio=0.6)

# pandas DataFrame
df = faker.generate_dataset(500, male_ratio=0.5, as_dataframe=True)
print(df['gender'].value_counts())
# male      250
# female    250
```

---

### `get_stats() -> Dict[str, int]`

Get statistics about the names database.

**Returns:** Dictionary with:
- `male_names_count`: Number of male first names
- `female_names_count`: Number of female first names
- `last_names_count`: Number of family names
- `total_names`: Sum of all names
- `possible_combinations`: Total possible combinations

**Example:**
```python
stats = faker.get_stats()
print(f"Possible combinations: {stats['possible_combinations']:,}")
# Possible combinations: 21,000,000
```

---

### Function: `generate_fake_name()`

```python
generate_fake_name(gender=None, seed=None) -> Dict[str, str]
```

Convenience function for quick one-off name generation.

**Example:**
```python
from farsi_faker import generate_fake_name

person = generate_fake_name('male', seed=42)
print(person['name'])
```

---

## 🎨 Examples

### Example 1: Create Test Dataset for Django

```python
from farsi_faker import FarsiFaker
from myapp.models import User

faker = FarsiFaker(seed=42)
dataset = faker.generate_dataset(100, male_ratio=0.5)

for person in dataset:
    User.objects.create(
        name=person['name'],
        first_name=person['first_name'],
        last_name=person['last_name'],
        gender=person['gender']
    )
```

### Example 2: Export to CSV

```python
import csv
from farsi_faker import FarsiFaker

faker = FarsiFaker()
dataset = faker.generate_dataset(1000, male_ratio=0.6)

with open('people.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'first_name', 'last_name', 'gender'])
    writer.writeheader()
    writer.writerows(dataset)
```

### Example 3: pandas DataFrame for Data Science

```python
import pandas as pd
from farsi_faker import FarsiFaker

faker = FarsiFaker(seed=123)

# Generate directly as DataFrame — no manual conversion needed
df = faker.generate_dataset(500, male_ratio=0.55, as_dataframe=True)

print(df.head())
print(df['gender'].value_counts())
print(df.describe(include='all'))

# Works with all standard pandas operations
grouped = df.groupby('gender')['last_name'].nunique()
print(grouped)
```

### Example 4: pytest Fixture

```python
import pytest
from farsi_faker import FarsiFaker

@pytest.fixture
def fake_users():
    faker = FarsiFaker(seed=42)
    return faker.generate_dataset(10, male_ratio=0.5)

def test_user_creation(fake_users):
    assert len(fake_users) == 10
    assert all('name' in user for user in fake_users)
```

### Example 5: API Mock Data

```python
from flask import Flask, jsonify
from farsi_faker import FarsiFaker

app = Flask(__name__)
faker = FarsiFaker()

@app.route('/api/users/random')
def random_user():
    return jsonify(faker.full_name())

@app.route('/api/users/<int:count>')
def multiple_users(count):
    users = faker.generate_names(min(count, 100))  # Max 100
    return jsonify(users)
```

---

## 🎯 Gender Input Options

The package accepts various gender formats:

### English
- `'male'`, `'m'` → Male
- `'female'`, `'f'` → Female

### Persian (فارسی)
- `'مرد'`, `'پسر'`, `'مذکر'` → Male  
- `'زن'`, `'دختر'`, `'مونث'` → Female

---

## 📊 Database Statistics

```python
from farsi_faker import FarsiFaker

faker = FarsiFaker()
stats = faker.get_stats()

print(f"Male names: {stats['male_names_count']:,}")
print(f"Female names: {stats['female_names_count']:,}")
print(f"Last names: {stats['last_names_count']:,}")
print(f"Total names: {stats['total_names']:,}")
print(f"Possible combinations: {stats['possible_combinations']:,}")
```

**Example Output:**
```
Male names: 3,500
Female names: 3,800
Last names: 2,700
Total names: 10,000
Possible combinations: 19,710,000
```

---

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=farsi_faker --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/alisadeghiaghili/farsi-faker.git
cd farsi-faker

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[all]"
```

### Code Quality

```bash
# Format code
black farsi_faker/
isort farsi_faker/

# Type checking
mypy farsi_faker/

# Run tests
pytest tests/ -v
```

### Building and Publishing

```bash
# Build distribution packages
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

---

## 📁 Project Structure

```
farsi-faker/
├── farsi_faker/              # Main package
│   ├── __init__.py           # Package initialization
│   ├── faker.py              # Core FarsiFaker class
│   ├── _version.py           # Version information
│   └── data/                 # Data directory
│       ├── __init__.py
│       └── names.pkl         # Pickle database (embedded)
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_faker.py
├── scripts/                  # Development scripts
│   └── create_pickle.py      # Build pickle from CSV
├── setup.py                  # Setup configuration
├── pyproject.toml            # Project metadata
├── MANIFEST.in               # Distribution files
├── LICENSE                   # MIT License
├── README.md                 # This file
└── CHANGELOG.md              # Version history
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run tests** (`pytest tests/`)
6. **Commit changes** (`git commit -m 'Add amazing feature'`)
7. **Push to branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Code Style

- Follow PEP 8
- Use Black for formatting
- Add type hints
- Write docstrings
- Add tests for new features

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Links

- **Author:** Ali Sadeghi Aghili
- **Email:** alisadeghiaghili@gmail.com
- **GitHub:** [https://github.com/alisadeghiaghili/farsi-faker](https://github.com/alisadeghiaghili/farsi-faker)
- **PyPI:** [https://pypi.org/project/farsi-faker/](https://pypi.org/project/farsi-faker/)
- **Issues:** [https://github.com/alisadeghiaghili/farsi-faker/issues](https://github.com/alisadeghiaghili/farsi-faker/issues)

---

## 🙏 Acknowledgments

- Names dataset sourced from publicly available Iranian name databases
- Inspired by [Faker](https://github.com/joke2k/faker) library
- Built with ❤️ for the Persian/Farsi development community

---

<div align="center">

Made with ❤️ by [Ali Sadeghi Aghili](https://github.com/alisadeghiaghili)

</div>
