"""Setup configuration for farsi-faker package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read version from _version.py
version_dict = {}
version_file = this_directory / "farsi_faker" / "_version.py"
with open(version_file, encoding="utf-8") as f:
    exec(f.read(), version_dict)

setup(
    name="farsi-faker",
    version=version_dict["__version__"],
    
    # Author information
    author="Ali Sadeghi Aghili",
    author_email="alisadeghiaghili@gmail.com",
    maintainer="Ali Sadeghi Aghili",
    maintainer_email="alisadeghiaghili@gmail.com",
    
    # Package description
    description="Generate realistic fake Persian/Farsi names for testing - تولید اسم‌های فارسی فیک با 10,000+ اسم اصیل",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # License
    license="MIT",
    
    # URLs
    url="https://github.com/alisadeghiaghili/farsi-faker",
    project_urls={
        "Homepage": "https://github.com/alisadeghiaghili/farsi-faker",
        "Documentation": "https://github.com/alisadeghiaghili/farsi-faker#readme",
        "Source Code": "https://github.com/alisadeghiaghili/farsi-faker",
        "Bug Tracker": "https://github.com/alisadeghiaghili/farsi-faker/issues",
        "Changelog": "https://github.com/alisadeghiaghili/farsi-faker/releases",
        "Download": "https://pypi.org/project/farsi-faker/",
    },
    
    # Package discovery
    packages=find_packages(
        exclude=[
            "tests",
            "tests.*",
            "scripts",
            "scripts.*",
            "data_sources",
            "data_sources.*",
            "docs",
            "examples",
        ]
    ),
    
    # Package data
    package_data={
        "farsi_faker": ["data/*.pkl"],
    },
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Dependencies
	# No dependencies! Zero-dependency package
    install_requires=[],  
    
    # Optional dependencies
    extras_require={
        # Development tools
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        # Data processing (only for creating pickle from CSV)
        "data": [
            "pandas>=1.3.0",
        ],
        # All optional dependencies
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "pandas>=1.3.0",
        ],
    },
    
    # PyPI classifiers
    classifiers=[
        # Development status
        "Development Status :: 5 - Production/Stable",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming language
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3 :: Only",
        
        # Natural language
        "Natural Language :: Persian",
        "Natural Language :: English",
        
        # Operating system
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Topics
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Testing :: Mocking",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Utilities",
        
        # Typing
        "Typing :: Typed",
    ],
    
    # Keywords for PyPI search (comma-separated string or list)
    keywords=[
        # Core
        "faker", "farsi", "persian", "iranian",
        "names", "fake-data", "test-data", "mock-data",
        
        # Language
        "farsi-names", "persian-names", "iranian-names",
        "parsi", "iran",
        
        # Functionality
        "name-generator", "fake-generator", "data-generator",
        "fake-names", "full-names", "first-names", "last-names",
        
        # Use cases
        "testing", "unittest", "pytest", "mock",
        "dataset", "synthetic-data", "sample-data",
        
        # Features
        "gender-specific", "reproducible", "seed-based",
        "no-dependencies", "lightweight", "fast",
        "zero-dependency",
        
        # Developer tools
        "development", "automation", "fixtures",
        
        # Localization
        "localization", "i18n", "l10n", "rtl",
        
        # Data science
        "data-science", "pandas",
        
        # Persian keywords
        "فارسی", "ایرانی", "پارسی",
    ],
    
    # Zip safe
    zip_safe=False,
)
