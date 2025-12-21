"""Data storage for farsi-faker names database.

Contains optimized pickle file with 10,000+ authentic Persian names.
This data is accessed automatically by FarsiFaker class.

Files:
    names.pkl: Optimized pickle database containing male names, female names,
               and family names from real Iranian datasets.

Note:
    This package is for internal use only. Users should not import from this
    package directly. Use the FarsiFaker class to access name data.

Example:
    >>> # Correct usage
    >>> from farsi_faker import FarsiFaker
    >>> faker = FarsiFaker()
    >>> 
    >>> # Incorrect usage (will raise AttributeError)
    >>> from farsi_faker.data import names  # Don't do this!
"""

__all__ = []


def __getattr__(name):
    """Prevent direct access to internal data files.
    
    This prevents users from directly importing the pickle file or other
    internal data. All data access should go through the FarsiFaker class
    for proper initialization and error handling.
    
    Args:
        name: Attribute name being accessed
    
    Raises:
        AttributeError: Always raised to prevent direct data access
    
    Example:
        >>> from farsi_faker.data import names
        AttributeError: 'names' should not be accessed directly.
        Use FarsiFaker class to access names data.
    """
    raise AttributeError(
        f"'{name}' should not be accessed directly. "
        "Use FarsiFaker class to access names data.\n\n"
        "Example:\n"
        "  from farsi_faker import FarsiFaker\n"
        "  faker = FarsiFaker()\n"
        "  person = faker.full_name()"
    )
