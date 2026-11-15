# Archive

This directory preserves material that is intentionally outside the canonical
v2 evidentiary boundary.

- `legacy_v1/` contains the complete earlier QSAR workflow and its outputs.
  It is the only archived tree kept in the public repository.

Non-evidentiary rendering scratch (`workspaces/`) and machine-local caches
(`local_state/`) were removed before public release and are gitignored so they
are not reintroduced. They were never confirmatory evidence.

Only `legacy_v1/` is a reproducible historical project. Its SHA-256 inventory
covers the git-tracked files in that snapshot; gitignored LaTeX auxiliaries,
`*.log` files, and `.DS_Store` entries are skipped so a clone verifies.
Verify it with:

```bash
python scripts/12_manage_archive_manifest.py
```
