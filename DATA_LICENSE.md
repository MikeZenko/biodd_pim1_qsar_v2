# Data license

Source code in this repository is licensed under the MIT License (`LICENSE`).
That grant does **not** cover ChEMBL-derived data.

## ChEMBL-derived files (CC BY-SA 3.0)

The following materials are derived from ChEMBL and are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-sa/3.0/):

- the pinned release snapshot under `data/releases/chembl37_pim1_20260501/`
- curated event, compound, lineage, and related tables under
  `data/curated/chembl37_pim1_v2/`
- immutable split files under `artifacts/split_locks/`
- any other file that contains ChEMBL records or fields derived from them,
  including prediction tables, assay-context extracts, and the legacy cohort
  files under `archive/legacy_v1/data/`

ShareAlike applies to adapted ChEMBL-derived tables: they must be distributed
under CC BY-SA 3.0 (or a compatible license) with the attribution below.

### Required attribution

ChEMBL data is from https://www.ebi.ac.uk/chembl — the version of ChEMBL is
chembl\_37.

### Required citation

Zdrazil, B. et al. The ChEMBL Database in 2023: a drug discovery platform
spanning multiple bioactivity data types and time periods. *Nucleic Acids
Research* **52**, D1180–D1192 (2024). https://doi.org/10.1093/nar/gkad1004

This is the citation already used in the manuscript
(`zdrazil2024chembl`). Additional ChEMBL terms are recorded in
`data/releases/chembl37_pim1_20260501/CHEMBL_README.txt`.

Third-party software and model checkpoints used at runtime, but not
redistributed here, are described in `NOTICE`.
