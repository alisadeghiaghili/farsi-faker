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

- **Python 3.9+**
- **No external dependencies** for production use
- Optional: `pandas` for DataFrame output (`as_dataframe=True`)

---

## 🚀 Quick Start

### Basic Usage

```python
from farsi_faker import FarsiFaker

faker = FarsiFaker()

# Generate a random person
person = faker.full_name()
print(person)
# {'name': 'علی صادقی عقیلی', 'first_name': 'علی', 'last_name': 'صادقی عقیلی', 'gender': 'male'}

# Generate male name
male = faker.full_name('male')
print(male['name'])   # علی صادقی عقیلی

# Generate female name
female = faker.full_name('female')
print(female['name'])  # سپیده جلیلی
```

### Generate Multiple Names

```python
# 10 random names as a list (default)
people = faker.generate_names(10)

# 50 male names as a list
men = faker.generate_names(50, 'male')

# 30 female names as a pandas DataFrame
women_df = faker.generate_names(30, 'female', as_dataframe=True)
print(women_df.shape)          # (30, 4)
print(list(women_df.columns))  # ['name', 'first_name', 'last_name', 'gender']
print(women_df.head(2))
#          name first_name last_name  gender
# 0  فاطمه احمدی     فاطمه    احمدی  female
# 1  زینب رضایی      زینب    رضایی  female
```

### Generate Balanced Dataset

```python
# 100 people with 60% male ratio — as a list
dataset = faker.generate_dataset(100, male_ratio=0.6)
print(len(dataset))   # 100

# Same, but as a pandas DataFrame
df = faker.generate_dataset(500, male_ratio=0.5, as_dataframe=True)
print(df.shape)                      # (500, 4)
print(df['gender'].value_counts())
# male      250
# female    250
# Name: gender, dtype: int64
```

### Reproducible Results

```python
faker1 = FarsiFaker(seed=42)
faker2 = FarsiFaker(seed=42)
assert faker1.full_name() == faker2.full_name()  # True
```

### Quick One-Off Generation

```python
from farsi_faker import generate_fake_name

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
faker = FarsiFaker()        # random
faker = FarsiFaker(seed=42) # reproducible
```

---

### `male_first_name() -> str`

Return a random male first name.

```python
faker.male_first_name()  # 'محمد'
```

---

### `female_first_name() -> str`

Return a random female first name.

```python
faker.female_first_name()  # 'فاطمه'
```

---

### `first_name(gender=None) -> Tuple[str, str]`

Return a first name with its normalised gender.

**Parameters:**
- `gender` (str, optional): Any supported gender token (see [Gender Input Options](#-gender-input-options))

**Returns:** `(name, gender)` — gender is always `'male'` or `'female'`

```python
name, g = faker.first_name('male')
# ('علی', 'male')

name, g = faker.first_name()   # random gender
# ('مریم', 'female')
```

---

### `last_name() -> str`

Return a random Persian family name.

```python
faker.last_name()  # 'احمدی'
```

---

### `full_name(gender=None) -> Dict[str, str]`

Return a complete person record.

**Returns:** dict with keys `name`, `first_name`, `last_name`, `gender`

```python
person = faker.full_name('female')
# {
#     'name': 'سپیده جلیلی',
#     'first_name': 'سپیده',
#     'last_name': 'جلیلی',
#     'gender': 'female'
# }
assert person['name'] == person['first_name'] + ' ' + person['last_name']
```

---

### `generate_names(count=10, gender=None, as_dataframe=False)`

Generate multiple full-name records.

**Parameters:**
- `count` (int, default `10`): Number of records to generate
- `gender` (str, optional): Gender applied to all records; random mix when `None`
- `as_dataframe` (bool, default `False`): Return a `pandas.DataFrame` instead of a list

**Returns:** `List[Dict]` or `pandas.DataFrame` with columns `['name', 'first_name', 'last_name', 'gender']`

**Raises:** `ValueError` if `count ≤ 0`; `ImportError` if `as_dataframe=True` and pandas is not installed

```python
# List (default)
people = faker.generate_names(5, 'male')
assert len(people) == 5
assert all(p['gender'] == 'male' for p in people)

# DataFrame
df = faker.generate_names(100, as_dataframe=True)
assert df.shape == (100, 4)
assert list(df.columns) == ['name', 'first_name', 'last_name', 'gender']
assert not df.isnull().any().any()
assert (df['name'] == df['first_name'] + ' ' + df['last_name']).all()
```

---

### `generate_dataset(count=100, male_ratio=0.5, as_dataframe=False)`

Generate a balanced dataset with a configurable gender ratio.

**Parameters:**
- `count` (int, default `100`): Total number of records
- `male_ratio` (float, default `0.5`): Fraction of male records in `[0.0, 1.0]`
- `as_dataframe` (bool, default `False`): Return a `pandas.DataFrame` instead of a list

**Returns:** Shuffled `List[Dict]` or `pandas.DataFrame`

**Raises:** `ValueError` if `count ≤ 0` or `male_ratio` outside `[0.0, 1.0]`; `ImportError` if pandas missing and `as_dataframe=True`

```python
# List (default)
dataset = faker.generate_dataset(10, male_ratio=0.6)
assert len(dataset) == 10
assert sum(1 for p in dataset if p['gender'] == 'male') == 6

# DataFrame
df = faker.generate_dataset(100, male_ratio=0.5, as_dataframe=True)
assert df.shape == (100, 4)
assert df['gender'].value_counts().to_dict() == {'male': 50, 'female': 50}

# Edge cases
assert all(p['gender'] == 'female' for p in faker.generate_dataset(5, male_ratio=0.0))
assert all(p['gender'] == 'male'   for p in faker.generate_dataset(5, male_ratio=1.0))
```

---

### `get_stats() -> Dict[str, int]`

Return statistics about the embedded names database.

**Returns:** dict with keys `male_names_count`, `female_names_count`, `last_names_count`, `total_names`, `possible_combinations`

```python
stats = faker.get_stats()
assert stats['possible_combinations'] == \
    (stats['male_names_count'] + stats['female_names_count']) * stats['last_names_count']
print(f"Possible combinations: {stats['possible_combinations']:,}")
# Possible combinations: 21,000,000
```

---

### Function: `generate_fake_name(gender=None, seed=None) -> Dict[str, str]`

Convenience wrapper for one-off generation. For bulk generation prefer a `FarsiFaker` instance directly.

```python
from farsi_faker import generate_fake_name

p1 = generate_fake_name('female', seed=99)
p2 = generate_fake_name('female', seed=99)
assert p1 == p2  # reproducible
```

---

## 🎨 Examples

### Example 1: Django test fixtures

```python
from farsi_faker import FarsiFaker
from myapp.models import User

faker = FarsiFaker(seed=42)
for person in faker.generate_dataset(100, male_ratio=0.5):
    User.objects.create(**person)
```

### Example 2: Export to CSV

```python
import csv
from farsi_faker import FarsiFaker

faker = FarsiFaker()
with open('people.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'first_name', 'last_name', 'gender'])
    writer.writeheader()
    writer.writerows(faker.generate_dataset(1000, male_ratio=0.6))
```

### Example 3: pandas DataFrame for data science

```python
from farsi_faker import FarsiFaker

faker = FarsiFaker(seed=123)
df = faker.generate_dataset(500, male_ratio=0.55, as_dataframe=True)

print(df.shape)                          # (500, 4)
print(df['gender'].value_counts())       # male 275 / female 225
print(df.groupby('gender')['last_name'].nunique())
```

### Example 4: pytest fixture

```python
import pytest
from farsi_faker import FarsiFaker

@pytest.fixture
def fake_users():
    return FarsiFaker(seed=42).generate_dataset(10, male_ratio=0.5)

def test_user_creation(fake_users):
    assert len(fake_users) == 10
    assert all('name' in u for u in fake_users)
```

### Example 5: Flask mock API

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
    return jsonify(faker.generate_names(min(count, 100)))
```

---

## 🎯 Gender Input Options

| Input | Resolves to |
|---|---|
| `'male'`, `'m'` | `'male'` |
| `'مرد'`, `'پسر'`, `'مذکر'` | `'male'` |
| `'female'`, `'f'` | `'female'` |
| `'زن'`, `'دختر'`, `'مونث'` | `'female'` |
| `None` | random |

---

## 📊 Database Statistics

```python
from farsi_faker import FarsiFaker

stats = FarsiFaker().get_stats()
print(f"Male names:            {stats['male_names_count']:,}")
print(f"Female names:          {stats['female_names_count']:,}")
print(f"Last names:            {stats['last_names_count']:,}")
print(f"Total names:           {stats['total_names']:,}")
print(f"Possible combinations: {stats['possible_combinations']:,}")
```

---

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ --cov=farsi_faker --cov-report=html
```

---

## 🛠️ Development

```bash
git clone https://github.com/alisadeghiaghili/farsi-faker.git
cd farsi-faker
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"

# quality checks
black farsi_faker/ && isort farsi_faker/ && mypy farsi_faker/
pytest tests/ -v
```

---

## 📁 Project Structure

```
farsi-faker/
├── farsi_faker/
│   ├── __init__.py
│   ├── faker.py          ← core class
│   ├── _version.py
│   └── data/names.pkl
├── tests/test_faker.py
├── scripts/create_pickle.py
├── pyproject.toml
├── CHANGELOG.md
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Run tests (`pytest tests/`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push and open a Pull Request

Code style: Black + isort. Type hints required. Docstrings required.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 📞 Contact

- **Author:** Ali Sadeghi Aghili
- **GitHub:** [alisadeghiaghili/farsi-faker](https://github.com/alisadeghiaghili/farsi-faker)
- **PyPI:** [pypi.org/project/farsi-faker](https://pypi.org/project/farsi-faker/)
- **Issues:** [github.com/alisadeghiaghili/farsi-faker/issues](https://github.com/alisadeghiaghili/farsi-faker/issues)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/alisadeghiaghili">Ali Sadeghi Aghili</a>
</div>
