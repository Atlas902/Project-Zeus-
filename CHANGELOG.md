# Changelog

## 0.3.0 — 2026-11-20

- Added validation for invalid and non-finite inputs.
- Made experience adjustments similarity-weighted instead of averaging unrelated records.
- Made confidence honest when there is no alternative action to compare.
- Preserved input order for tied decisions.
- Prevented simulator actions after a terminal goal or hazard state.
- Added grid validation and clearer memory corruption errors.
- Fixed `.gitignore` newline formatting and excluded generated experience logs.
- Expanded automated coverage from 5 to 21 tests.
