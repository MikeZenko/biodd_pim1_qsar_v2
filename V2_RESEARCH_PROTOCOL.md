# PIM1 QSAR publication protocol (v2)

This protocol defines the confirmatory replacement for the legacy 3,110-compound
analysis. Legacy files and results remain in the repository for provenance, but
they are not confirmatory evidence for the v2 manuscript.

## Frozen source

- Target: human PIM1 (`CHEMBL2147`).
- Database: ChEMBL 37, release date 2026-05-01.
- Snapshot ID: `chembl37_pim1_20260501`.
- The acquisition stage refuses a mismatched ChEMBL database version or release
  date and stores official release notes, checksums, deterministic gzip JSONL
  tables, and a SHA-256 manifest.
- Structures are standardized with `chembl-structure-pipeline==1.2.4`; package
  substitution is forbidden.

## Cohorts

The broad cohort requires exact positive IC50 values in nM, `standard_flag=1`,
valid structures, no ChEMBL duplicate flag, and no data-validity warning.

The primary strict cohort additionally requires:

- biochemical assay type (`B`);
- single-protein BAO format (`BAO_0000357`);
- direct assay-to-target relationship (`D`);
- confidence score 9;
- no assay variant accession or mutation.

Repeated source events are collapsed before compound medians are calculated,
while every activity ID remains in the event-lineage table. Prespecified
sensitivity cohorts remove pIC50 values above 10 and assay `CHEMBL3705754`.

## Outcome-blind split lock

The temporal cutoff is selected using counts only. The locked rule chooses the
latest year with at least 750 development compounds, 250 later compounds, and
100 new later scaffolds. It selects 2018.

- Development: first eligible record by 2018; labels use only records through
  2018.
- Sealed temporal test: first eligible record in 2019 or later.
- Later records for development identities are quarantined for label-drift
  analysis and never update development labels.
- Development validation: three repeated five-fold stratified scaffold-group
  splits, seeds 1101-1103.
- Inner tuning: three scaffold-group folds created separately inside each outer
  training partition.

The immutable split lock is
`artifacts/split_locks/chembl37_strict_temporal2018_scaffold3x5_v1`.

## Prespecified candidate comparison

Every model consumes the same stored outer assignments. Selection minimizes the
mean of per-repeat pooled out-of-fold RMSE values.

- Baselines: development mean; Tanimoto k-nearest neighbors on ECFP6.
- Classical: Ridge, Random Forest (ECFP6 and combined), Gradient Boosting, and
  RBF-SVR.
- Graph: edge-aware GINE and Chemprop D-MPNN, each graph-only and graph plus the
  same 23 RDKit descriptors.

ECFP6 uses radius 3 and 2,048 bits. Learned preprocessing is fitted inside each
inner-training fold. GNN hyperparameters are fixed across outer folds; three
inner scaffold folds select epoch counts, after which a fresh model is trained
on the complete outer-training partition for the median selected epoch count.

## Final evaluation policy

Candidate ranking may not read `temporal_test_sealed.csv`. Only the selected
model is refit on all development data and evaluated on the temporal table.
Primary uncertainty uses a 10,000-replicate Bemis-Murcko scaffold-cluster
bootstrap. Additional diagnostics include a development-defined five-neighbor
Tanimoto applicability domain, calibration, and a cross-validated absolute-
residual interval. The latter is descriptive under temporal shift and is not
claimed as a formal exchangeability-based coverage guarantee.

## Reproduction order

```text
scripts/01_build_snapshot.py
scripts/02_curate_snapshot.py
scripts/03_lock_splits.py
scripts/04_run_classical_nested.py
scripts/05_run_gnn_nested.py
scripts/06_select_development_model.py
scripts/07_evaluate_temporal_once.py
```

All computational training runs on the Google Cloud VM. The local workspace is
used only for source editing, small artifact synchronization, manuscript
generation, and visual QA.
