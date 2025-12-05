# Data Directory

This folder contains data related to the **KR3 Korean restaurant sentiment** project.

The actual dataset is not committed to this repository due to size and licensing.
Instead, we load the data directly from the Hugging Face Hub:

- Dataset ID: `leey4n/KR3`
- Columns:
  - `Review`: the original Korean restaurant review text
  - `Rating`: label (0, 1, 2).
    - `0`: negative
    - `1`: positive
    - `2`: ambiguous (we drop this label in our experiments)

## Structure

```text
data/
├─ README.md        # this file
├─ raw/             # optional: raw dumps or cached downloads
└─ processed/       # optional: pre-processed / filtered versions
```
