# API Reference | مرجع API

Complete API documentation for Farsi-Faker package.

مستندات کامل API برای پکیج فارسی-فیکر

Version: 1.0.0 | نسخه: ۱.۰.۰

---

## Table of Contents | فهرست مطالب

- [FarsiFaker Class | کلاس FarsiFaker](#farsifaker-class--کلاس-farsifaker)
- [Module Functions | توابع ماژول](#module-functions--توابع-ماژول)
- [Type Definitions | تعریف انواع](#type-definitions--تعریف-انواع)
- [Exceptions | استثناها](#exceptions--استثناها)

---

## FarsiFaker Class | کلاس FarsiFaker

**English:** Main class for generating Persian/Farsi names.

**فارسی:** کلاس اصلی برای تولید اسم‌های فارسی/پارسی.

### Constructor | سازنده

```python
FarsiFaker(seed: Optional[int] = None)
```

**English:** Creates a new instance of the FarsiFaker class.

**فارسی:** یک نمونه جدید از کلاس FarsiFaker ایجاد می‌کند.

**Parameters | پارامترها:**

| Name<br>نام | Type<br>نوع | Default<br>پیش‌فرض | Description (English)<br>توضیحات (فارسی) |
|------|------|---------|-------------|
| seed | `Optional[int]` | `None` | Random seed for reproducible results. If None, uses system randomness.<br>بذر تصادفی برای نتایج قابل تکرار. اگر None باشد، از تصادفی‌سازی سیستم استفاده می‌کند. |

**Example | مثال:**

```python
from farsi_faker import FarsiFaker

# Random generation | تولید تصادفی
faker = FarsiFaker()

# Reproducible generation | تولید قابل تکرار
faker = FarsiFaker(seed=42)
```

---

### Instance Methods | متدهای نمونه

#### `male_first_name()` | اسم کوچک مردانه

```python
def male_first_name(self) -> str
```

**English:** Generate a random male first name.

**فارسی:** یک اسم کوچک مردانه تصادفی تولید می‌کند.

**Returns | برگشتی:**
- `str`: A randomly selected authentic male Persian name
- `str`: یک اسم مردانه اصیل فارسی به صورت تصادفی

**Example | مثال:**

```python
name = faker.male_first_name()
# Returns | برمی‌گرداند: 'محمد'
```

---

#### `female_first_name()` | اسم کوچک زنانه

```python
def female_first_name(self) -> str
```

**English:** Generate a random female first name.

**فارسی:** یک اسم کوچک زنانه تصادفی تولید می‌کند.

**Returns | برگشتی:**
- `str`: A randomly selected authentic female Persian name
- `str`: یک اسم زنانه اصیل فارسی به صورت تصادفی

**Example | مثال:**

```python
name = faker.female_first_name()
# Returns | برمی‌گرداند: 'فاطمه'
```

---

#### `first_name()` | اسم کوچک

```python
def first_name(self, gender: Optional[str] = None) -> Tuple[str, str]
```

**English:** Generate a first name with optional gender specification.

**فارسی:** یک اسم کوچک با مشخص کردن جنسیت (اختیاری) تولید می‌کند.

**Parameters | پارامترها:**

| Name<br>نام | Type<br>نوع | Default<br>پیش‌فرض | Description (English)<br>توضیحات (فارسی) |
|------|------|---------|-------------|
| gender | `Optional[str]` | `None` | Gender specification. Accepts: 'male', 'female', 'm', 'f', 'مرد', 'زن', 'پسر', 'دختر', 'مذکر', 'مونث'<br>مشخص کردن جنسیت. قبول می‌کند: 'male', 'female', 'm', 'f', 'مرد', 'زن', 'پسر', 'دختر', 'مذکر', 'مونث' |

**Returns | برگشتی:**
- `Tuple[str, str]`: (name, normalized_gender) where gender is 'male' or 'female'
- `Tuple[str, str]`: (اسم، جنسیت نرمال‌شده) که جنسیت 'male' یا 'female' است

**Raises | خطاها:**
- `ValueError`: If gender value is not recognized
- `ValueError`: اگر مقدار جنسیت شناخته نشود

**Example | مثال:**

```python
# Specific gender | جنسیت مشخص
name, gender = faker.first_name('male')
# Returns | برمی‌گرداند: ('علی', 'male')

# Random gender | جنسیت تصادفی
name, gender = faker.first_name()
# Returns | برمی‌گرداند: ('مریم', 'female')

# Persian input | ورودی فارسی
name, gender = faker.first_name('زن')
# Returns | برمی‌گرداند: ('فاطمه', 'female')
```

---

#### `last_name()` | نام خانوادگی

```python
def last_name(self) -> str
```

**English:** Generate a random family name.

**فارسی:** یک نام خانوادگی تصادفی تولید می‌کند.

**Returns | برگشتی:**
- `str`: A randomly selected authentic Persian family name
- `str`: یک نام خانوادگی اصیل فارسی به صورت تصادفی

**Example | مثال:**

```python
name = faker.last_name()
# Returns | برمی‌گرداند: 'احمدی'
```

---

#### `full_name()` | نام کامل

```python
def full_name(self, gender: Optional[str] = None) -> Dict[str, str]
```

**English:** Generate a complete person with full name and metadata.

**فارسی:** یک شخص کامل با نام کامل و اطلاعات جانبی تولید می‌کند.

**Parameters | پارامترها:**

| Name<br>نام | Type<br>نوع | Default<br>پیش‌فرض | Description (English)<br>توضیحات (فارسی) |
|------|------|---------|-------------|
| gender | `Optional[str]` | `None` | Desired gender. Accepts same formats as first_name()<br>جنسیت مورد نظر. همان فرمت‌های first_name() را قبول می‌کند |

**Returns | برگشتی:**
- `Dict[str, str]`: Dictionary with keys: `name`, `first_name`, `last_name`, `gender`
- `Dict[str, str]`: دیکشنری با کلیدهای: `name` (نام کامل)، `first_name` (نام)، `last_name` (نام خانوادگی)، `gender` (جنسیت)

**Dictionary Keys | کلیدهای دیکشنری:**
- `name`: Full name (first + last) | نام کامل (نام + نام خانوادگی)
- `first_name`: First name only | فقط نام کوچک
- `last_name`: Family name only | فقط نام خانوادگی
- `gender`: Normalized gender ('male' or 'female') | جنسیت نرمال‌شده ('male' یا 'female')

**Raises | خطاها:**
- `ValueError`: If gender is invalid | اگر جنسیت نامعتبر باشد

**Example | مثال:**

```python
person = faker.full_name('female')
# Returns | برمی‌گرداند:
# {
#     'name': 'فاطمه محمدی',
#     'first_name': 'فاطمه',
#     'last_name': 'محمدی',
#     'gender': 'female'
# }
```

---

#### `generate_names()` | تولید چند اسم

```python
def generate_names(
    self,
    count: int = 10,
    gender: Optional[str] = None
) -> List[Dict[str, str]]
```

**English:** Generate multiple full names.

**فارسی:** چندین نام کامل تولید می‌کند.

**Parameters | پارامترها:**

| Name<br>نام | Type<br>نوع | Default<br>پیش‌فرض | Description (English)<br>توضیحات (فارسی) |
|------|------|---------|-------------|
| count | `int` | `10` | Number of names to generate (must be positive)<br>تعداد اسم‌هایی که باید تولید شوند (باید مثبت باشد) |
| gender | `Optional[str]` | `None` | Gender for all names. If None, randomly mixes genders<br>جنسیت برای همه اسم‌ها. اگر None باشد، جنسیت‌ها تصادفی ترکیب می‌شوند |

**Returns | برگشتی:**
- `List[Dict[str, str]]`: List of person dictionaries (see full_name() for structure)
- `List[Dict[str, str]]`: لیست دیکشنری‌های شخص (برای ساختار به full_name() مراجعه کنید)

**Raises | خطاها:**
- `ValueError`: If count is not positive or gender is invalid
- `ValueError`: اگر count مثبت نباشد یا جنسیت نامعتبر باشد

**Example | مثال:**

```python
# Generate 5 male names | تولید ۵ اسم مردانه
men = faker.generate_names(5, 'male')

# Generate 10 random gender names | تولید ۱۰ اسم با جنسیت تصادفی
people = faker.generate_names(10)

# All with Persian gender input | همه با ورودی جنسیت فارسی
women = faker.generate_names(20, 'زن')
```

---

#### `generate_dataset()` | تولید مجموعه داده

```python
def generate_dataset(
    self,
    count: int = 100,
    male_ratio: float = 0.5
) -> List[Dict[str, str]]
```

**English:** Generate a balanced dataset with specified gender ratio.

**فارسی:** یک مجموعه داده متعادل با نسبت جنسیتی مشخص تولید می‌کند.

**Parameters | پارامترها:**

| Name<br>نام | Type<br>نوع | Default<br>پیش‌فرض | Description (English)<br>توضیحات (فارسی) |
|------|------|---------|-------------|
| count | `int` | `100` | Total number of names to generate (must be positive)<br>تعداد کل اسم‌ها (باید مثبت باشد) |
| male_ratio | `float` | `0.5` | Ratio of male names (0.0 to 1.0). Examples: 0.5=balanced, 0.7=70% male, 1.0=all male<br>نسبت اسم‌های مردانه (۰.۰ تا ۱.۰). مثال: ۰.۵=متعادل، ۰.۷=۷۰٪ مرد، ۱.۰=همه مرد |

**Returns | برگشتی:**
- `List[Dict[str, str]]`: List of person dictionaries in random order (shuffled)
- `List[Dict[str, str]]`: لیست دیکشنری‌های شخص به ترتیب تصادفی (مخلوط‌شده)

**Raises | خطاها:**
- `ValueError`: If count is not positive or male_ratio is out of range
- `ValueError`: اگر count مثبت نباشد یا male_ratio خارج از محدوده باشد

**Example | مثال:**

```python
# 60% male, 40% female | ۶۰٪ مرد، ۴۰٪ زن
dataset = faker.generate_dataset(100, male_ratio=0.6)

# All female | همه زن
all_women = faker.generate_dataset(50, male_ratio=0.0)

# Balanced (default) | متعادل (پیش‌فرض)
balanced = faker.generate_dataset(100)

# All male | همه مرد
all_men = faker.generate_dataset(50, male_ratio=1.0)
```

---

#### `get_stats()` | گرفتن آمار

```python
def get_stats(self) -> Dict[str, int]
```

**English:** Get statistics about the names database.

**فارسی:** آمار پایگاه داده اسم‌ها را برمی‌گرداند.

**Returns | برگشتی:**
- `Dict[str, int]`: Dictionary with database statistics
- `Dict[str, int]`: دیکشنری حاوی آمار پایگاه داده

**Dictionary Keys | کلیدهای دیکشنری:**
- `male_names_count`: Number of unique male first names | تعداد اسم‌های کوچک مردانه یکتا
- `female_names_count`: Number of unique female first names | تعداد اسم‌های کوچک زنانه یکتا
- `last_names_count`: Number of unique family names | تعداد نام‌های خانوادگی یکتا
- `total_names`: Sum of all unique names | مجموع همه اسم‌های یکتا
- `possible_combinations`: Total possible full name combinations | کل ترکیبات ممکن نام کامل

**Example | مثال:**

```python
stats = faker.get_stats()
print(f"Male names: {stats['male_names_count']:,}")
print(f"Female names: {stats['female_names_count']:,}")
print(f"Possible combinations: {stats['possible_combinations']:,}")

# Returns | برمی‌گرداند:
# {
#     'male_names_count': 3500,         # اسم‌های مردانه: ۳۵۰۰
#     'female_names_count': 3800,       # اسم‌های زنانه: ۳۸۰۰
#     'last_names_count': 2700,         # نام خانوادگی: ۲۷۰۰
#     'total_names': 10000,             # کل اسم‌ها: ۱۰۰۰۰
#     'possible_combinations': 19710000 # ترکیبات ممکن: ۱۹،۷۱۰،۰۰۰
# }
```

---

## Module Functions | توابع ماژول

### `generate_fake_name()` | تولید اسم فیک

```python
def generate_fake_name(
    gender: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, str]
```

**English:** Quick function to generate a single fake Persian name. Convenience function that creates a FarsiFaker instance and generates one name. For generating multiple names, create a FarsiFaker instance directly for better performance.

**فارسی:** تابع سریع برای تولید یک اسم فیک فارسی. تابع راحتی که یک نمونه FarsiFaker ایجاد و یک اسم تولید می‌کند. برای تولید چند اسم، مستقیماً یک نمونه FarsiFaker ایجاد کنید (کارایی بهتر).

**Parameters | پارامترها:**

| Name<br>نام | Type<br>نوع | Default<br>پیش‌فرض | Description (English)<br>توضیحات (فارسی) |
|------|------|---------|-------------|
| gender | `Optional[str]` | `None` | Desired gender (same formats as FarsiFaker)<br>جنسیت مورد نظر (همان فرمت‌های FarsiFaker) |
| seed | `Optional[int]` | `None` | Random seed for reproducibility<br>بذر تصادفی برای قابلیت تکرار |

**Returns | برگشتی:**
- `Dict[str, str]`: Person dictionary with full name and metadata
- `Dict[str, str]`: دیکشنری شخص با نام کامل و اطلاعات جانبی

**Example | مثال:**

```python
from farsi_faker import generate_fake_name

# Quick male name | اسم مردانه سریع
person = generate_fake_name('male')

# Quick female name | اسم زنانه سریع
person = generate_fake_name('زن')

# Reproducible | قابل تکرار
person1 = generate_fake_name('female', seed=123)
person2 = generate_fake_name('female', seed=123)
assert person1 == person2  # True | درست
```

---

## Type Definitions | تعریف انواع

### GenderType | نوع جنسیت

```python
GenderType = Literal['male', 'female']
```

**English:** Normalized gender type (internal use).

**فارسی:** نوع جنسیت نرمال‌شده (استفاده داخلی).

---

### GenderInput | ورودی جنسیت

```python
GenderInput = Union[str, None]
```

**English:** Flexible gender input type that accepts various formats.

**فارسی:** نوع ورودی انعطاف‌پذیر جنسیت که فرمت‌های مختلف را قبول می‌کند.

**Valid Values | مقادیر معتبر:**

| Format<br>فرمت | Values<br>مقادیر | Meaning<br>معنی |
|---------|---------|---------|
| English | `'male'`, `'m'` | Male / مرد |
| English | `'female'`, `'f'` | Female / زن |
| Persian | `'مرد'`, `'پسر'`, `'مذکر'` | Male / مرد |
| Persian | `'زن'`, `'دختر'`, `'مونث'` | Female / زن |
| Special | `None` | Random selection / انتخاب تصادفی |

**Notes | نکات:**
- Case-insensitive | بدون حساسیت به بزرگ/کوچک بودن حروف
- Strips whitespace | فاصله‌های اضافی حذف می‌شوند

---

## Exceptions | استثناها

### ValueError | خطای مقدار

**English:** Raised when invalid parameters are provided to methods.

**فارسی:** وقتی پارامترهای نامعتبر به متدها داده می‌شود، رخ می‌دهد.

**Common Cases | موارد رایج:**

#### 1. Invalid Gender | جنسیت نامعتبر

```python
faker.full_name('invalid')
# ValueError: Invalid gender: 'invalid'
# Valid values: 'male', 'female', 'm', 'f', 'مرد', 'زن', ...

# فارسی: خطای مقدار: جنسیت نامعتبر: 'invalid'
# مقادیر معتبر: 'male', 'female', 'm', 'f', 'مرد', 'زن', ...
```

#### 2. Invalid Count | تعداد نامعتبر

```python
faker.generate_names(0)
# ValueError: Count must be positive, got: 0
# فارسی: خطای مقدار: تعداد باید مثبت باشد، دریافت شد: ۰

faker.generate_names(-5)
# ValueError: Count must be positive, got: -5
# فارسی: خطای مقدار: تعداد باید مثبت باشد، دریافت شد: ‎-۵
```

#### 3. Invalid Ratio | نسبت نامعتبر

```python
faker.generate_dataset(100, male_ratio=1.5)
# ValueError: male_ratio must be between 0 and 1, got: 1.5
# Examples: 0.5 (balanced), 0.7 (70% male), 1.0 (all male)

# فارسی: خطای مقدار: male_ratio باید بین ۰ و ۱ باشد، دریافت شد: ۱.۵
# مثال‌ها: ۰.۵ (متعادل)، ۰.۷ (۷۰٪ مرد)، ۱.۰ (همه مرد)
```

---

### FileNotFoundError | خطای فایل پیدا نشد

**English:** Raised when the names data file cannot be found.

**فارسی:** وقتی فایل داده اسم‌ها پیدا نمی‌شود، رخ می‌دهد.

```python
# Raised during FarsiFaker initialization if data file is missing
# در هنگام ساخت FarsiFaker اگر فایل داده موجود نباشد رخ می‌دهد

faker = FarsiFaker()
# FileNotFoundError: Names data file not found: ...
# Please ensure the package is installed correctly.
# Try reinstalling: pip install --force-reinstall farsi-faker

# فارسی: خطای فایل پیدا نشد: فایل داده اسم‌ها پیدا نشد: ...
# لطفاً مطمئن شوید پکیج به درستی نصب شده است.
# نصب مجدد را امتحان کنید: pip install --force-reinstall farsi-faker
```

---

### pickle.UnpicklingError | خطای باز کردن pickle

**English:** Raised when the data file is corrupted.

**فارسی:** وقتی فایل داده خراب شده باشد، رخ می‌دهد.

```python
# Raised during FarsiFaker initialization if pickle is corrupted
# در هنگام ساخت FarsiFaker اگر فایل pickle خراب باشد رخ می‌دهد

faker = FarsiFaker()
# pickle.UnpicklingError: Failed to load names data: ...
# The data file may be corrupted. Try reinstalling the package:
# pip install --force-reinstall farsi-faker

# فارسی: خطای باز کردن pickle: بارگذاری داده اسم‌ها شکست خورد: ...
# فایل داده ممکن است خراب شده باشد. نصب مجدد پکیج را امتحان کنید:
# pip install --force-reinstall farsi-faker
```

---

## Version Information | اطلاعات نسخه

**English:** Access version information programmatically.

**فارسی:** دسترسی برنامه‌نویسی به اطلاعات نسخه.

```python
from farsi_faker import __version__, __version_info__

print(__version__)       # '1.0.0'
print(__version_info__)  # (1, 0, 0)

# Get full version info | گرفتن اطلاعات کامل نسخه
from farsi_faker import get_info
info = get_info()
print(info['version'])   # '1.0.0'
print(info['author'])    # 'Ali Sadeghi Aghili'
```

---

## Performance Characteristics | مشخصات کارایی

### Time Complexity | پیچیدگی زمانی

| Method<br>متد | Complexity<br>پیچیدگی | Description<br>توضیحات |
|---------|------------|-------------|
| `male_first_name()` | O(1) | Constant time / زمان ثابت |
| `female_first_name()` | O(1) | Constant time / زمان ثابت |
| `last_name()` | O(1) | Constant time / زمان ثابت |
| `first_name()` | O(1) | Constant time / زمان ثابت |
| `full_name()` | O(1) | Constant time / زمان ثابت |
| `generate_names(n)` | O(n) | Linear time / زمان خطی |
| `generate_dataset(n)` | O(n log n) | Due to shuffling / به خاطر مخلوط‌کردن |

### Space Complexity | پیچیدگی فضایی

**English:**
- FarsiFaker instance: O(1) - uses cached data
- Data cache: O(n) where n is total number of names (~10,000)

**فارسی:**
- نمونه FarsiFaker: O(1) - از داده حافظه‌پنهان استفاده می‌کند
- حافظه‌پنهان داده: O(n) که n تعداد کل اسم‌هاست (~۱۰,۰۰۰)

### Initialization | راه‌اندازی اولیه

**English:**
- First instance: ~10-50ms (loads pickle)
- Subsequent instances: <1ms (uses cache)

**فارسی:**
- اولین نمونه: ~۱۰-۵۰ میلی‌ثانیه (بارگذاری pickle)
- نمونه‌های بعدی: <۱ میلی‌ثانیه (استفاده از حافظه‌پنهان)

---

## Thread Safety | امنیت چندنخی

**English:** FarsiFaker is thread-safe:
- Data is loaded once and shared across all instances
- Each instance has its own random generator
- No shared mutable state between calls

**فارسی:** FarsiFaker ایمن برای چندنخی است:
- داده یکبار بارگذاری و بین همه نمونه‌ها مشترک می‌شود
- هر نمونه تولیدکننده تصادفی خودش را دارد
- هیچ حالت تغییرپذیر مشترکی بین فراخوانی‌ها نیست

**Example | مثال:**

```python
from concurrent.futures import ThreadPoolExecutor
from farsi_faker import FarsiFaker

def generate_name(seed):
    faker = FarsiFaker(seed=seed)
    return faker.full_name()

# Safe to use in multiple threads | استفاده ایمن در چند نخ
with ThreadPoolExecutor(max_workers=10) as executor:
    names = list(executor.map(generate_name, range(100)))
```

---

## Best Practices | بهترین روش‌ها

### 1. Reuse Instances | استفاده مجدد از نمونه‌ها

**English:** For generating multiple names, reuse the same FarsiFaker instance.

**فارسی:** برای تولید چند اسم، از همان نمونه FarsiFaker استفاده مجدد کنید.

```python
# Good | خوب ✅
faker = FarsiFaker()
names = [faker.full_name() for _ in range(1000)]

# Less efficient | کمتر کارآمد ❌
names = [FarsiFaker().full_name() for _ in range(1000)]
```

---

### 2. Use Seeds for Testing | استفاده از بذر برای تست

**English:** Use seeds in tests for reproducible results.

**فارسی:** در تست‌ها از بذر برای نتایج قابل تکرار استفاده کنید.

```python
import pytest
from farsi_faker import FarsiFaker

@pytest.fixture
def faker():
    return FarsiFaker(seed=42)

def test_name_generation(faker):
    name = faker.full_name()
    # Deterministic | قطعی
    assert name == {'name': '...', ...}
```

---

### 3. Use generate_dataset() for Balanced Data | استفاده از generate_dataset() برای داده متعادل

**English:** For balanced datasets, use `generate_dataset()` instead of multiple calls.

**فارسی:** برای مجموعه‌داده‌های متعادل، از `generate_dataset()` به جای فراخوانی‌های متعدد استفاده کنید.

```python
# Good | خوب ✅
dataset = faker.generate_dataset(1000, male_ratio=0.5)

# Less efficient | کمتر کارآمد ❌
males = faker.generate_names(500, 'male')
females = faker.generate_names(500, 'female')
dataset = males + females
random.shuffle(dataset)
```

---

### 4. Handle Exceptions Properly | مدیریت صحیح استثناها

```python
from farsi_faker import FarsiFaker

try:
    faker = FarsiFaker()
    person = faker.full_name('male')
except FileNotFoundError:
    print("Package not installed correctly")
    print("پکیج به درستی نصب نشده است")
except ValueError as e:
    print(f"Invalid input: {e}")
    print(f"ورودی نامعتبر: {e}")
```

---

## Usage Examples | مثال‌های کاربردی

### Example 1: CSV Export | مثال ۱: خروجی CSV

```python
import csv
from farsi_faker import FarsiFaker

# Create faker | ایجاد faker
faker = FarsiFaker(seed=42)

# Generate dataset | تولید مجموعه داده
dataset = faker.generate_dataset(1000, male_ratio=0.6)

# Export to CSV | خروجی به CSV
with open('people.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['name', 'first_name', 'last_name', 'gender'])
    writer.writeheader()
    writer.writerows(dataset)

print("✅ CSV file created successfully!")
print("✅ فایل CSV با موفقیت ایجاد شد!")
```

---

### Example 2: Django Models | مثال ۲: مدل‌های جنگو

```python
from farsi_faker import FarsiFaker
from myapp.models import User

# Create faker | ایجاد faker
faker = FarsiFaker(seed=123)

# Generate test users | تولید کاربران تست
dataset = faker.generate_dataset(100, male_ratio=0.5)

for person in dataset:
    User.objects.create(
        name=person['name'],
        first_name=person['first_name'],
        last_name=person['last_name'],
        gender=person['gender']
    )

print("✅ 100 test users created!")
print("✅ ۱۰۰ کاربر تست ایجاد شد!")
```

---

### Example 3: pandas DataFrame | مثال ۳: دیتافریم پانداس

```python
import pandas as pd
from farsi_faker import FarsiFaker

# Create faker | ایجاد faker
faker = FarsiFaker(seed=42)

# Generate dataset | تولید مجموعه داده
dataset = faker.generate_dataset(500, male_ratio=0.55)

# Create DataFrame | ایجاد دیتافریم
df = pd.DataFrame(dataset)

# Statistics | آمار
print("First 10 rows | ۱۰ ردیف اول:")
print(df.head(10))

print("
Gender distribution | توزیع جنسیت:")
print(df['gender'].value_counts())

print("
Dataset info | اطلاعات مجموعه داده:")
print(df.info())
```

---

## Quick Reference | مرجع سریع

### Most Common Operations | رایج‌ترین عملیات

```python
from farsi_faker import FarsiFaker, generate_fake_name

# Create instance | ایجاد نمونه
faker = FarsiFaker(seed=42)

# Single name | یک اسم
person = faker.full_name('male')
# {'name': 'علی احمدی', ...}

# Multiple names | چند اسم
people = faker.generate_names(10, 'female')

# Balanced dataset | مجموعه داده متعادل
dataset = faker.generate_dataset(100, male_ratio=0.6)

# Quick one-off | سریع یکبار مصرف
person = generate_fake_name('زن', seed=123)

# Statistics | آمار
stats = faker.get_stats()
```

---

## Support & Contact | پشتیبانی و تماس

**English:**
- GitHub: https://github.com/alisadeghiaghili/farsi-faker
- Issues: https://github.com/alisadeghiaghili/farsi-faker/issues
- Email: alisadeghiaghili@gmail.com

**فارسی:**
- گیت‌هاب: https://github.com/alisadeghiaghili/farsi-faker
- گزارش مشکلات: https://github.com/alisadeghiaghili/farsi-faker/issues
- ایمیل: alisadeghiaghili@gmail.com

---

For more information, see [README.md](README.md) and [examples](examples/).

برای اطلاعات بیشتر، [README.md](README.md) و [examples](examples/) را ببینید.
