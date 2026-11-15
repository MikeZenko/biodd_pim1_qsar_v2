# PIM1 QSAR Benchmark v2

This repository contains a release-pinned, temporally sealed benchmark for
human PIM1 biochemical IC50 prediction. Its primary purpose is to test temporal
generalization and prediction-level reliability under joint chemical, cohort,
and assay-provenance shift.
A bounded comparison of fixed ECFP6/descriptor representations with edge-aware
GINE and Chemprop D-MPNN models is secondary; all candidates use identical
scaffold-disjoint folds.

An expanded graph comparison is also provided as a versioned, explicitly post
hoc development-only sensitivity. It performs nested architecture and optimizer
search for GINE and Chemprop D-MPNN, quantifies initialization variability with
three-seed ensembles, and adds a frozen CheMeleon pretrained representation. It
does not reopen or revise the completed temporal evaluation.

The completed expanded graph sensitivity found:

- Frozen CheMeleon plus a nested-tuned head: repeated scaffold-CV RMSE
  0.754 +/- 0.011 pIC50
- Nested-tuned edge-aware GINE: RMSE 0.760 +/- 0.012
- Nested-tuned Chemprop D-MPNN: RMSE 0.779 +/- 0.011
- Best graph-family paired RMSE difference versus the locked Ridge OOF
  reference: +0.002 (95% scaffold-cluster CI -0.038 to +0.046)
- Complete search inventory: 1,350 trial objects, 4,050 inner-fold fits,
  45 outer selections, and three fresh outer-refit seeds per selection

The expanded search therefore repairs the comparison but does not establish a
graph-model advantage over Ridge. It remains post hoc and development-only.

The frozen primary v2 result is:

- ChEMBL 37 strict cohort: 1,125 measurement events, 1,087 standardized compounds
- Development partition: 803 compounds through 2018
- Sealed temporal partition: 284 compounds first meeting the strict cohort
  definition in 2019--2025
- Development-selected model: Ridge on ECFP6 plus 23 RDKit descriptors
- Repeated scaffold-CV RMSE: 0.742 pIC50
- One-time temporal RMSE: 0.972 (95% scaffold-cluster CI 0.872--1.075)
- One-time temporal R2: 0.470 (95% CI 0.311--0.582)

The explicitly post hoc year-aware audit adds:

- Post hoc annual RMSE: 0.795 in 2019 and 1.261 in 2024; difference 0.466
  (95% scaffold-cluster CI 0.190--0.733)
- Post hoc 2020--2025 within-year-centered R2: -0.046; Spearman rho: 0.286

The post hoc robustness audit adds:

- Document-source-set sensitivity CI: RMSE 0.787--1.150 and R2 -0.007--0.682
- Within-document centered ordering (groups with at least three compounds):
  R2 -0.031; Spearman rho 0.303
- Four temporal identities had earlier broad but non-strict records; excluding
  them gave RMSE 0.976 and R2 0.463
- Nearest-neighbor potency discordance correlated with absolute error:
  Spearman rho 0.271 (95% scaffold-cluster CI 0.122--0.411)

The post hoc assay/checkpoint audits add:

- Strict cohort provenance: 122 assays in 102 documents; the development and
  temporal partitions share zero assay and zero document identifiers
- Numeric ATP context: 117/1,125 events (10.4%), spanning 20 uM--5 mM
- Assay identifiability: only 18 compounds cross assays, while 82.2% of events
  belong to scaffolds seen in a single assay
- GNN epoch choices: none of 180 inner selected epochs or 60 assigned outer
  refit durations equaled the 80-epoch ceiling (maxima 63 and 43, respectively)

The prespecified reliability diagnostics showed:

- Tanimoto-domain error correlation: Spearman rho 0.042
- Development-residual reference band: 2.504 pIC50 wide, 81.3% temporal
  coverage when defined at a 90% development target

The year-aware quantities are explicitly post hoc descriptive audits of the
immutable one-time temporal predictions. They were not used to choose, tune, or
refit the model and are not retroactively part of the frozen protocol.

The residual reference band is not a formal conformal predictor: its scores
come from fold-specific OOF models after development-based candidate selection,
while its centers come from a separate final refit. Historical sealed fields
prefixed `conformal_` retain their original schema names for compatibility; they
do not denote a formally valid conformal construction.

These results concern retrospective biochemical potency. They do not establish
PIM-family selectivity, cellular activity, ADMET behavior, safety, efficacy, or
prospective discovery performance.

## Start here

- `V2_RESEARCH_PROTOCOL.md`: frozen scientific protocol
- `ARTIFACT_INDEX.md`: canonical artifacts and legacy boundary
- `docs/PROJECT_STRUCTURE.md`: current layout and archive policy
- `manuscript_v2/main.pdf`: revised paper
- `manuscript_v2/generated/manuscript_results.json`: machine-readable paper values
- `artifacts/runs/temporal_evaluation_v2/temporal_evaluation_manifest.json`:
  one-time evaluation record and checksums
- `artifacts/audits/temporal_robustness_v2/audit_manifest.json`: post hoc audit
  inputs, outputs, seeds, and checksums

## Reproducible v2 workflow

All model fitting is intended for a VM. The two environments are deliberately
separate.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/setup_gnn_vm.sh
```

The production stages are:

```bash
.venv/bin/python scripts/00_capture_environment.py \
  --label classical-curation --requirements requirements.txt \
  --output artifacts/environment/classical_curation.json

.venv/bin/python scripts/01_build_snapshot.py
.venv/bin/python scripts/02_curate_snapshot.py
.venv/bin/python scripts/03_lock_splits.py
.venv/bin/python scripts/04_run_classical_nested.py --n-jobs 10
.venv-gnn/bin/python scripts/05_run_gnn_nested.py \
  --max-epochs 80 --patience 10 --hidden-dim 128 --torch-threads 4
.venv/bin/python scripts/06_select_development_model.py
.venv/bin/python scripts/07_evaluate_temporal_once.py --n-jobs 10
.venv/bin/python scripts/10_build_posthoc_robustness_audit.py
.venv-gnn/bin/python scripts/11_run_gnn_tuned_nested.py \
  --family edge_aware_gine --repeat 0 --fold 0
# Repeat the preceding command for all 3 families, 3 repeats, and 5 folds.
.venv-gnn/bin/python scripts/11_run_gnn_tuned_nested.py --aggregate-only
.venv/bin/python scripts/08_build_manuscript_assets.py
python scripts/09_seal_manuscript.py
```

`07_evaluate_temporal_once.py` is guarded: when a completed evaluation already
exists, it verifies hashes and exits without reopening the sealed label table.
Do not delete or replace the completed temporal directory to manufacture a new
"one-time" result.

## Validation design

- Strict direct, wild-type, single-protein human PIM1 biochemical assays
- ChEMBL confidence score 9 and direct relationship type D
- Parent standardization with `chembl-structure-pipeline==1.2.4`
- Outcome-blind 2018 cutoff selected from counts only
- Post-cutoff records for development identities quarantined from labels
- Three repeats of five-fold stratified scaffold-group outer validation
- Three scaffold-group inner folds for tuning or epoch selection
- Post hoc graph sensitivity: one family-specific anchored baseline plus 23
  deterministic random draws, with the same candidate set evaluated in every
  outer fold, top-three reranking over three seeds, and three-seed outer
  ensembles
- Shared identity and fold inventories across all eleven candidates
- RMSE selected on development out-of-fold predictions only
- 10,000-replicate Bemis--Murcko scaffold-cluster bootstrap
- Training-defined Tanimoto applicability domain and descriptive residual reference intervals

## Project layout

```text
configs/protocol_v2.yaml                  Machine-readable locked protocol
data/releases/chembl37_pim1_20260501/    Release snapshot and checksums
data/curated/chembl37_pim1_v2/           Curated events, compounds, lineage
artifacts/split_locks/                    Immutable time/scaffold assignments
artifacts/runs/classical_nested_v2/       Classical development benchmark
artifacts/runs/gnn_nested_v2/             GNN development benchmark
artifacts/runs/gnn_tuned_posthoc_v1/       Expanded development-only graph search
artifacts/runs/development_selection_v2/  Development-only selection
artifacts/runs/temporal_evaluation_v2/    One-time result and fitted model
artifacts/audits/temporal_robustness_v2/  Post hoc robustness and source audit
manuscript_v2/                            Revised paper, figures, generated tables
src/pim1_qsar/                            Reusable curation, validation, model code
tests/v2/                                 Unit and scientific-contract tests
archive/legacy_v1/                        Preserved pre-v2 workflow and outputs
```

## Tests

```bash
.venv/bin/pytest -q tests/v2
.venv-gnn/bin/pytest -q \
  tests/v2/test_gnn_feature_contract.py \
  tests/v2/test_gnn_nested_contract.py \
  tests/v2/test_temporal_evaluation_contract.py
```

Final validation produced 96 passing v2 tests in the local classical
environment (three optional-dependency skips) and 21 passing targeted tests in
the VM GNN environment (one expected skip). After public-release packaging,
the classical suite reports 98 passed and 2 optional-dependency skips when
Chemprop/Torch are absent. The suite includes pinned
assay-context, robustness-audit, GNN-duration, tuned-search lineage, exact
trial/selection inventory, output-hash, and independently recomputed metric
checks.

## Legacy boundary

The complete pre-v2 workflow is preserved under `archive/legacy_v1/`, including
its numbered scripts, paper, source data, tests, serialized models, results,
plots, and logs. It uses an older 3,110-compound cohort and a reused scaffold
holdout. It is not confirmatory evidence for the revised manuscript. Canonical
v2 results live under `artifacts/` and `manuscript_v2/`. Run
`python scripts/12_manage_archive_manifest.py` to verify the archived snapshot.

## Citation and release

Code and confirmatory artifacts are available in the public GitHub repository
and an archival Zenodo snapshot:

- GitHub: `https://github.com/OWNER/biodd_pim1_qsar_v2`
  (TODO: replace OWNER with the public repository owner)
- Zenodo: `doi:10.5281/zenodo.XXXXXXX`
  (TODO: replace with the deposited Zenodo DOI)

ChEMBL-derived tables are redistributed under CC BY-SA 3.0 with attribution to
ChEMBL 37 (Zdrazil et al., 2024).

Until a journal version is available, cite the manuscript title:

> Narynov, T. M. (2026). A Release-Pinned and Temporally Sealed PIM1 QSAR
> Benchmark: Temporal Reliability under Chemical and Cohort Shift.
> ORCID: https://orcid.org/0009-0006-9003-415X

## License

Source code is released under the MIT License (`LICENSE`).

ChEMBL-derived data (the release snapshot, curated tables, split files, and
any files containing ChEMBL records) are licensed under Creative Commons
Attribution-ShareAlike 3.0 Unported. See `DATA_LICENSE.md`.

Third-party notices, including Chemprop and the CheMeleon pretrained checkpoint
(used by content hash from Zenodo and not redistributed here), are in `NOTICE`.
