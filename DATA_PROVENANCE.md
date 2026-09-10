# Data Provenance — farsi-faker

This document describes how the embedded name database
(`farsi_faker/data/names.pkl`) is produced and what guarantees the test
suite enforces on every release.

## Source status

| Item | Status |
|------|--------|
| Original CSV sources (`data_sources/iranianNamesDataset.csv`, `iranian-surname-frequencies.csv`) | **Not in the repository** (removed before v1.0.0 public history was curated) |
| Rebuild path from raw CSVs | **Unavailable** until sources are restored |
| Current artifact | Deterministically cleaned from the previously shipped pickle |

Until raw sources are restored, the **authoritative** rebuild input is the
last released `names.pkl`, processed by `farsi_faker.cleaning` /
`scripts/rebuild_names_pkl.py`.

## Pipeline (as of v1.4.0)

1. Load `names.pkl` (`male_names`, `female_names`, `last_names`).
2. `precision_repair()` per name:
   - `join_ocr_splits` — singleton tokens (`آ رمان`) and short-head splits (`آر مان`)
   - `join_abdol_family` — glue `عبد` / `عب` compounds (`عبد الله` → `عبدالله`)
   - `normalize_zwnj` — strip ZWNJ–space hybrids and edge ZWNJ
   - drop truncated `ال` endings and bare prefixes (`عبد`, `آق`, …)
   - drop names with fewer than 3 letters after compaction
   - drop gender-label noise (female honorifics in the male pool, …)
3. Resolve male/female overlaps (keep female side).
4. Sort, unique, write pickle (protocol HIGHEST).

Rebuild command:

```bash
python scripts/rebuild_names_pkl.py --dry-run
python scripts/rebuild_names_pkl.py
```

## Pool sizes by release

| Release | male | female | last | notes |
|---------|-----:|-------:|-----:|-------|
| ≤1.1.1 | 7863 | 3817 | 5755 | pre-cleaning |
| 1.2.0 | 7633 | 3730 | 5755 | OCR singleton + honorific pass |
| 1.3.0 | 7493 | 3648 | 5748 | Abdol glue + truncated tokens |
| 1.4.0 | 7493 | 3648 | 5748 | ZWNJ hygiene (no size change expected) |

Exact numbers after each rebuild are printed by `rebuild_names_pkl.py`.

## CI gates

`tests/test_data_quality.py` fails the build if any of the following
reappear in the embedded pickle:

- singleton-token OCR artifacts
- female honorifics in the male pool / male titles leading female names
- gender pool overlap
- unsorted or duplicate pools
- empty / double-space names
- truncated `ال` endings
- unglued bare `عبد` / `عب`
- names shorter than 3 letters

## Profile field provenance

Synthetic contact fields (`national_id`, `mobile`, `email`, `postal_code`,
`city`, `street`, `alley`, `plaque`) are **generated**, not scraped from
real people:

- National ID uses the official Iranian 10-digit checksum.
- Mobile numbers use real operator *prefixes* with random suffixes.
- Emails romanize generated Persian names.
- Cities/street labels are drawn from a built-in list of major Iranian
  cities and common street names.

They are for test fixtures only.
