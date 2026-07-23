# API Reference

```python
from farsi_faker import FarsiFaker, generate_fake_name

faker = FarsiFaker(seed=42)  # omit seed for random output
```

---

## Name generation

### `male_first_name() -> str`

Return a random male Persian first name.

```python
faker.male_first_name()  # 'محمد'
```

---

### `female_first_name() -> str`

Return a random female Persian first name.

```python
faker.female_first_name()  # 'فاطمه'
```

---

### `first_name(gender=None) -> Tuple[str, str]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gender` | `str \| None` | `None` | Gender token — see [Gender tokens](#gender-tokens). `None` picks randomly. |

**Returns:** `(first_name: str, gender: Literal["male", "female"])`

**Raises:** `ValueError` if the gender token is not recognised.

```python
name, g = faker.first_name('male')  # ('علی', 'male')
name, g = faker.first_name('زن')   # ('فاطمه', 'female')
name, g = faker.first_name()        # random gender
```

---

### `last_name() -> str`

Return a random Persian family name.

```python
faker.last_name()  # 'احمدی'
```

---

### `full_name(gender=None) -> Dict[str, str]`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gender` | `str \| None` | `None` | Gender token. `None` picks randomly. |

**Returns:** `{"name": str, "first_name": str, "last_name": str, "gender": str}`

**Raises:** `ValueError` if gender is invalid.

```python
person = faker.full_name('female')
# {
#   'name': 'سپیده جلیلی',
#   'first_name': 'سپیده',
#   'last_name': 'جلیلی',
#   'gender': 'female'
# }
assert person['name'] == person['first_name'] + ' ' + person['last_name']
```

---

### `generate_names(count=10, gender=None, as_dataframe=False)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `count` | `int` | `10` | Number of records. Must be a positive integer (booleans rejected). |
| `gender` | `str \| None` | `None` | Applied to all records. `None` gives a random mix. |
| `as_dataframe` | `bool` | `False` | Return a `pandas.DataFrame` instead of a list. |

**Returns:** `List[Dict[str, str]]` **or** `pandas.DataFrame` with columns `["name", "first_name", "last_name", "gender"]`.

**Raises:**
- `ValueError` — `count` is not a positive integer.
- `ImportError` — `as_dataframe=True` but pandas is not installed. The error message includes the correct install command.

```python
people = faker.generate_names(5, 'male')
assert all(p['gender'] == 'male' for p in people)

df = faker.generate_names(100, as_dataframe=True)
assert df.shape == (100, 4)
assert list(df.columns) == ['name', 'first_name', 'last_name', 'gender']
```

---

### `generate_dataset(count=100, male_ratio=0.5, as_dataframe=False)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `count` | `int` | `100` | Total records. Must be a positive integer (booleans rejected). |
| `male_ratio` | `float` | `0.5` | Fraction of male records. Must be a finite number in `[0.0, 1.0]`. |
| `as_dataframe` | `bool` | `False` | Return a `pandas.DataFrame` instead of a list. |

**Returns:** Shuffled `List[Dict[str, str]]` **or** `pandas.DataFrame`.

**Raises:**
- `ValueError` — `count ≤ 0` or `male_ratio` outside `[0.0, 1.0]`.
- `ImportError` — `as_dataframe=True` but pandas is not installed.

```python
dataset = faker.generate_dataset(10, male_ratio=0.6)
assert sum(1 for p in dataset if p['gender'] == 'male') == 6

# Edge cases
assert all(p['gender'] == 'female' for p in faker.generate_dataset(5, male_ratio=0.0))
assert all(p['gender'] == 'male'   for p in faker.generate_dataset(5, male_ratio=1.0))

# DataFrame
df = faker.generate_dataset(100, male_ratio=0.5, as_dataframe=True)
assert df['gender'].value_counts().to_dict() == {'male': 50, 'female': 50}
```

---

### `get_stats() -> Dict[str, int]`

Return statistics about the embedded names database.

**Returns:** dict with keys:
- `male_names_count` — unique male first names
- `female_names_count` — unique female first names
- `last_names_count` — unique family names
- `total_names` — sum of all unique names
- `possible_combinations` — `(male + female) × last_names`

```python
stats = faker.get_stats()
print(f"Possible combinations: {stats['possible_combinations']:,}")
# Possible combinations: 21,000,000
```

---

### `generate_fake_name(gender=None, seed=None) -> Dict[str, str]`

Module-level convenience function. Creates a temporary `FarsiFaker(seed)` and returns one `full_name()` record. For bulk generation use a persistent `FarsiFaker` instance instead.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gender` | `str \| None` | `None` | Gender token. |
| `seed` | `int \| None` | `None` | Reproducibility seed. |

```python
from farsi_faker import generate_fake_name

p1 = generate_fake_name('female', seed=99)
p2 = generate_fake_name('female', seed=99)
assert p1 == p2
```

---

## Synthetic profile generation

> ⚠️ **All values in this section are synthetic test data only.**
> Do **not** use them for authentication, identity verification, real communications,
> or any production purpose. `national_id()` produces checksum-valid values but
> does not guarantee the value is unassigned.

### `mobile_number() -> str`

Return a synthetic Iranian-format mobile number (e.g. `09123456789`).

```python
faker.mobile_number()  # '09123456789'
```

---

### `email() -> str`

Return a non-deliverable email using the reserved `example.test` domain.

```python
faker.email()  # 'ali.ahmadi@example.test'
```

---

### `national_id() -> str`

Return a ten-digit string that passes the Iranian national ID checksum algorithm.

> ⚠️ **For test data only.** This value may coincide with a real national ID.
> Never use it for authentication, KYC, or any verification purpose.

```python
faker.national_id()  # '0012345678'
```

---

### `postal_code() -> str`

Return a ten-digit synthetic postal code.

```python
faker.postal_code()  # '1234567890'
```

---

### `address() -> Dict[str, str]`

Return a synthetic address record.

**Returns:** `{"province": str, "city": str, "street_address": str, "postal_code": str}`

```python
addr = faker.address()
# {
#   'province': 'تهران',
#   'city': 'تهران',
#   'street_address': 'خیابان ولیعصر، پلاک ۱۲',
#   'postal_code': '1234567890'
# }
```

---

### `full_profile(gender=None) -> Dict[str, str]`

Return a complete synthetic profile combining name and address fields.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gender` | `str \| None` | `None` | Gender token. `None` picks randomly. |

**Returns:** Name record keys (`name`, `first_name`, `last_name`, `gender`) **plus** `mobile_number`, `email`, `national_id`, `postal_code`, `province`, `city`, `street_address`.

> ⚠️ **Test data only** — see warning at the top of this section.

```python
profile = faker.full_profile('male')
print(profile['name'])          # 'علی احمدی'
print(profile['mobile_number']) # '09123456789'
print(profile['national_id'])   # '0012345678'
```

---

## Gender tokens

| Token | Resolves to |
|---|---|
| `'male'`, `'m'` | `'male'` |
| `'مرد'`, `'پسر'`, `'مذکر'` | `'male'` |
| `'female'`, `'f'` | `'female'` |
| `'زن'`, `'دختر'`, `'مونث'` | `'female'` |
| `None` | random |

Gender tokens are case-insensitive and leading/trailing whitespace is stripped.

---

## Reproducibility and thread safety

The same `seed` produces the same sequence **within a single ordered call chain**.
Shared `FarsiFaker` instances are thread-safe, but output ordering across
concurrently scheduled calls is scheduling-dependent and not deterministic.

```python
# Safe: one instance shared across threads
faker = FarsiFaker()

# Each thread gets its own reproducible sequence
def worker(seed):
    return FarsiFaker(seed=seed).full_name()
```
