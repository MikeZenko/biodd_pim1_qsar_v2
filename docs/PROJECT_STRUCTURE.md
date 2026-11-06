# Project structure

The repository root is reserved for the active, release-pinned v2 benchmark.
Historical work is preserved, but it is physically separated so that old
holdout results cannot be mistaken for evidence from the temporal study.

## Active v2 study

- `configs/`: locked scientific and graph-search contracts.
- `data/releases/`: immutable ChEMBL release snapshot.
- `data/curated/`: curated v2 events, compounds, and lineage.
- `src/pim1_qsar/`: reusable implementation.
- `scripts/`: ordered production, audit, manuscript, and integrity entrypoints.
- `tests/v2/`: unit, lineage, leakage, and completed-artifact checks.
- `artifacts/`: immutable split locks, completed runs, environments, and audits.
- `manuscript_v2/`: sealed source, generated assets, figures, and final PDF.

The canonical evidentiary boundary is defined in `ARTIFACT_INDEX.md`.

## Preserved history

- `archive/legacy_v1/`: the complete earlier broad-cohort workflow. Scripts,
  input CSVs, tests, paper sources, models, results, plots, and logs remain
  together so their original relative paths still resolve when run from that
  directory.
- `archive/workspaces/`: non-canonical rendering and inspection scratch.
- `archive/local_state/`: machine-generated caches and editor/session files.

No historical files were discarded during the July 2026 reorganization. The
legacy snapshot is protected by `archive/legacy_v1/ARCHIVE_MANIFEST.json`.
Verify it with:

```bash
python scripts/12_manage_archive_manifest.py
```

Regenerate the inventory only after an intentional, documented archive change:

```bash
python scripts/12_manage_archive_manifest.py --write
```

## Placement policy

New scientific code belongs under `src/` with an entrypoint under `scripts/`.
New generated evidence belongs under a versioned `artifacts/runs/` or
`artifacts/audits/` directory with a manifest and completion marker. Manuscript
numbers must be generated under `manuscript_v2/generated/`; they should not be
copied manually from exploratory notebooks or archived v1 results.

Temporary renders and diagnostics belong outside the active evidence tree. Do
not place new model outputs, plots, or manuscript drafts at repository root.
