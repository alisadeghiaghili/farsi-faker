"""Persian → Latin romanization.

A character-level transliterator used to render Persian names as readable
Latin text — for an English-style output (``fullname_en``) and as the single
shared transliterator behind ``email_address``.

It is deliberately simple and **lossy**: Persian's inherent short vowels
(i, u, e) are not written in the source text, so they cannot be recovered.
What it reliably fixes over the old ad-hoc email table is the consonant
mapping that table got wrong (ح→h not s, خ→kh, ق/غ→gh, چ→ch, ش→sh, ژ→zh,
پ→p, گ→g, ک→k, ع→a), and it maps Persian / Arabic-Indic digits to ASCII and
drops ZWNJ.

The map is **identity for already-ASCII input** (so a Latin seed such as
``"Ali"`` passes through as ``"ali"``), which keeps Latin-seeded inputs
stable across the package.
"""

from __future__ import annotations

import re

__all__ = ["to_latin"]

# Persian (and Arabic) letters -> Latin, readability-first for names.
# و is treated as the vowel it most often is in names (o/ou); at the head of a
# word it is consonantal, but the character map cannot distinguish the two —
# an accepted limitation of character-level transliteration.
_FA_TO_LATIN = str.maketrans(
    {
        'ا': 'a',
        'آ': 'a',
        'ب': 'b',
        'پ': 'p',
        'ت': 't',
        'ث': 's',
        'ج': 'j',
        'چ': 'ch',
        'ح': 'h',
        'خ': 'kh',
        'د': 'd',
        'ذ': 'z',
        'ر': 'r',
        'ز': 'z',
        'ژ': 'zh',
        'س': 's',
        'ش': 'sh',
        'ص': 's',
        'ض': 'z',
        'ط': 't',
        'ظ': 'z',
        'ع': 'a',
        'غ': 'gh',
        'ف': 'f',
        'ق': 'gh',
        'ک': 'k',
        'گ': 'g',
        'ل': 'l',
        'م': 'm',
        'ن': 'n',
        'و': 'o',
        'ه': 'h',
        'ی': 'i',
        'ئ': 'i',
        'ء': '',
        '‌': '',  # ZWNJ
        # Persian / Arabic-Indic digits -> ASCII
        '۰': '0',
        '۱': '1',
        '۲': '2',
        '۳': '3',
        '۴': '4',
        '۵': '5',
        '۶': '6',
        '۷': '7',
        '۸': '8',
        '۹': '9',
        '٠': '0',
        '١': '1',
        '٢': '2',
        '٣': '3',
        '٤': '4',
        '٥': '5',
        '٦': '6',
        '٧': '7',
        '٨': '8',
        '٩': '9',
    }
)

_NON_LATIN = re.compile(r'[^a-z0-9]+')


def to_latin(text: str) -> str:
    """Transliterate Persian text to readable lowercase Latin.

    Runs of characters with no Latin equivalent (including unmapped letters,
    punctuation, and spaces) are collapsed to a single ``-`` and trimmed.

    Args:
        text (str): Persian (or already-Latin) text.

    Returns:
        str: Lowercase ``[a-z0-9-]`` string; ``''`` when nothing transliterates.

    Example:
        >>> to_latin('احمدی')
        'ahmdi'
        >>> to_latin('علی')
        'ali'
        >>> to_latin('حسینی')
        'hsini'
        >>> to_latin('Ali')  # already-Latin input is identity modulo case
        'ali'
        >>> to_latin('')  # nothing to transliterate
        ''
    """
    if not text:
        return ''
    lowered = text.translate(_FA_TO_LATIN).lower()
    cleaned = _NON_LATIN.sub('-', lowered).strip('-')
    return re.sub(r'-{2,}', '-', cleaned)
