# Farsi Faker | فارسی فیکر

<div align="center">

[![PyPI version](https://badge.fury.io/py/farsi-faker.svg)](https://pypi.org/project/farsi-faker/)
[![Python Support](https://img.shields.io/pypi/pyversions/farsi-faker.svg)](https://pypi.org/project/farsi-faker/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/farsi-faker)](https://pepy.tech/project/farsi-faker)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install farsi-faker
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
- Optional: `pandas` for data processing (development only)

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
# {'name': 'علی احمدی', 'first_name': 'علی', 'last_name': 'احمدی', 'gender': 'male'}

# Generate male name
male = faker.full_name('male')
print(male['name'])  # محمد رضایی

# Generate female name
female = faker.full_name('female')
print(female['name'])  # فاطمه محمدی
```

### Generate Multiple Names

```python
# Generate 10 random names
people = faker.generate_names(10)

# Generate 50 male names
men = faker.generate_names(50, 'male')

# Generate 30 female names
women = faker.generate_names(30, 'female')
```

### Generate Balanced Dataset

```python
# Generate 100 people with 60% male ratio
dataset = faker.generate_dataset(100, male_ratio=0.6)

# Verify ratio
males = sum(1 for p in dataset if p['gender'] == 'male')
print(f"Males: {males}, Females: {100 - males}")
# Males: 60, Females: 40
```

---

## 📖 Documentation

For complete API documentation, please visit our [documentation page](https://alisadeghiaghili.github.io/farsi-faker/).

### Quick Reference

```python
from farsi_faker import FarsiFaker

faker = FarsiFaker(seed=42)  # Optional seed for reproducibility

# Generate names
person = faker.full_name()              # Random gender
male = faker.full_name('male')          # Male name
female = faker.full_name('female')      # Female name

# Generate multiple
people = faker.generate_names(10)                    # 10 random
dataset = faker.generate_dataset(100, male_ratio=0.6)  # Balanced dataset

# Get statistics
stats = faker.get_stats()
print(stats['possible_combinations'])  # 21,000,000+
```

---

## 🎨 Examples

### Django Models

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

### Export to CSV

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

### pandas DataFrame

```python
import pandas as pd
from farsi_faker import FarsiFaker

faker = FarsiFaker(seed=123)
dataset = faker.generate_dataset(500, male_ratio=0.55)

df = pd.DataFrame(dataset)
print(df.head())
print(df['gender'].value_counts())
```

---

## 🎯 Gender Input Options

The package accepts various gender formats:

**English:** `'male'`, `'m'`, `'female'`, `'f'`

**Persian (فارسی):** `'مرد'`, `'پسر'`, `'مذکر'`, `'زن'`, `'دختر'`, `'مونث'`

---

## 🧪 Testing

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=farsi_faker --cov-report=html
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run tests (`pytest tests/`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

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
- **Website:** [https://alisadeghiaghili.github.io/farsi-faker/](https://alisadeghiaghili.github.io/farsi-faker/)
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

⭐ If you find this project useful, please consider giving it a star! ⭐

</div>

---

<div align="center">

Made with ❤️ by [Ali Sadeghi Aghili](https://github.com/alisadeghiaghili)

</div>