"""Data-quality gates for the embedded names database.

These tests run against the shipped ``names.pkl`` and fail CI if OCR
artifacts or gender-label noise reappear.
"""

from __future__ import annotations

from farsi_faker import FarsiFaker
from farsi_faker.cleaning import drop_from_pool, has_singleton_token


def _pools():
    faker = FarsiFaker()
    return faker._male_names, faker._female_names, faker._last_names


class TestEmbeddedDataQuality:
    """Invariants that must hold for every release."""

    def test_pools_are_non_empty(self) -> None:
        male, female, last = _pools()
        assert len(male) >= 100
        assert len(female) >= 100
        assert len(last) >= 100

    def test_no_singleton_token_artifacts(self) -> None:
        male, female, last = _pools()
        for label, pool in (
            ('male', male),
            ('female', female),
            ('last', last),
        ):
            bad = [name for name in pool if has_singleton_token(name)]
            assert bad == [], f'{label} pool still has OCR singleton tokens: {bad[:10]}'

    def test_no_female_honorifics_in_male_pool(self) -> None:
        male, _, _ = _pools()
        bad = [name for name in male if drop_from_pool(name, pool_gender='male')]
        assert bad == [], f'male pool honorific noise: {bad[:10]}'

    def test_no_male_titles_leading_female_pool(self) -> None:
        _, female, _ = _pools()
        bad = [name for name in female if drop_from_pool(name, pool_gender='female')]
        assert bad == [], f'female pool title noise: {bad[:10]}'

    def test_male_and_female_pools_are_disjoint(self) -> None:
        male, female, _ = _pools()
        overlap = set(male) & set(female)
        assert overlap == set(), f'gender overlap: {sorted(overlap)[:10]}'

    def test_pools_are_unique_and_sorted(self) -> None:
        for pool in _pools():
            assert list(pool) == sorted(set(pool))

    def test_no_empty_or_whitespace_names(self) -> None:
        for pool in _pools():
            for name in pool:
                assert name.strip() != ''
                assert '  ' not in name

    def test_repair_examples_present_in_clean_data(self) -> None:
        """Known OCR splits must not survive; their repairs should exist."""
        male, female, _ = _pools()
        male_set = set(male)
        female_set = set(female)

        # These forms were present in the pre-1.2.0 database.
        for artifact in ('آ رمان', 'ا میر', 'با با', 'ر حما ن'):
            assert artifact not in male_set
            assert artifact not in female_set

        assert 'آرمان' in male_set
        assert 'امیر' in male_set

    def test_core_common_names_present(self) -> None:
        """The most frequent Iranian names must ship in the embedded pool."""
        male, female, last = _pools()
        for name in ('محمد', 'علی', 'حسین', 'رضا', 'مهدی', 'مرتضی', 'یوسف'):
            assert name in male, f'missing male core name: {name}'
        for name in ('فاطمه', 'زهرا', 'مریم', 'زینب', 'نرگس', 'معصومه'):
            assert name in female, f'missing female core name: {name}'
        assert 'محمدی' in last

    def test_initial_coverage_for_meem_and_feh(self) -> None:
        """Pools must not be missing common Persian initials م and ف."""
        male, female, _ = _pools()
        assert sum(1 for n in male if n.startswith('م')) >= 10
        assert sum(1 for n in male if n.startswith('ف')) >= 5
        assert sum(1 for n in female if n.startswith('م')) >= 10
        assert sum(1 for n in female if n.startswith('ف')) >= 5

    def test_no_truncated_al_tokens(self) -> None:
        from farsi_faker.cleaning import is_truncated_name

        for pool in _pools():
            bad = [name for name in pool if is_truncated_name(name)]
            assert bad == [], f'truncated ال names remain: {bad[:10]}'

    def test_abdol_family_is_glued(self) -> None:
        """عبد-family names must not keep a bare عبد/عب head token."""
        male, female, _ = _pools()
        for pool in (male, female):
            bad = [
                name
                for name in pool
                if name == 'عبد'
                or name == 'عب'
                or name.startswith('عبد ')
                or name.startswith('عب ')
            ]
            assert bad == [], f'unglued Abdol names: {bad[:10]}'

        # Reconstructed forms should exist for common compounds.
        assert 'عبدالله' in set(male) | set(female) or any(
            'عبدالله' in name for name in male
        )

    def test_no_tiny_names(self) -> None:
        for pool in _pools():
            for name in pool:
                compact = name.replace(' ', '')
                assert len(compact) >= 3, f'tiny name: {name!r}'
