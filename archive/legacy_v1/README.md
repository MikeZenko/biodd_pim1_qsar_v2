# Legacy v1 snapshot

This is the preserved pre-v2 PIM1 QSAR workflow. It uses the older broad
3,110-compound cohort, partial nesting, and a reused scaffold holdout. Its
results are development history, not confirmatory evidence for the v2 temporal
benchmark or revised manuscript.

The original layout has been kept intact inside this directory: numbered
pipeline scripts, supporting modules, `data/`, `models/`, `results/`, `plots/`,
`figs/`, paper sources, tests, and logs are colocated so relative paths continue
to resolve.

For historical inspection only:

```bash
cd archive/legacy_v1
python -m venv .venv
.venv/bin/pip install -r requirements.txt
ALLOW_LIVE_REFRESH=1 .venv/bin/python 1.Download_Bioactivity_Data.py
```

The live-refresh workflow can change source data and must never overwrite or be
combined with `data/releases/`, `data/curated/`, `artifacts/`, or
`manuscript_v2/` in the repository root.

The complete file inventory and SHA-256 checksums are recorded in
`ARCHIVE_MANIFEST.json` and verified from the repository root with:

```bash
python scripts/12_manage_archive_manifest.py
```
