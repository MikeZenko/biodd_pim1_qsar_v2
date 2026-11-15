# Confirmatory v2 Artifact Index

This file defines the evidential boundary for the revised PIM1 manuscript.
Anything listed under **Canonical v2** may support confirmatory claims. Files
listed under **Legacy** are retained for history and must not be blended into v2
tables, figures, or conclusions.

## Canonical v2

### Environments

- `artifacts/environment/classical_curation.json`
- `artifacts/environment/gnn.json`

### Release snapshot and curation

- `data/releases/chembl37_pim1_20260501/snapshot_manifest.json`
- `data/curated/chembl37_pim1_v2/curation_manifest.json`
- `data/curated/chembl37_pim1_v2/strict_measurement_events.csv`
- `data/curated/chembl37_pim1_v2/compound_strict.csv`
- `data/curated/chembl37_pim1_v2/event_lineage.csv`
- `data/curated/chembl37_pim1_v2/exclusion_log.csv`
- `data/curated/chembl37_pim1_v2/strict_measurement_dispersion.csv`

Primary curation identity:
`3d4b301bb42c911056ace75ac03d4db12335e5c8ef8bfed95e8fe74a1789bd05`.

### Immutable split

- `artifacts/split_locks/chembl37_strict_temporal2018_scaffold3x5_v1/`
- Lock SHA-256:
  `80b4c761afdc693a01ff9a6c408ecfa5e4d5811d3bbab7b58447c7cdd1295634`

### Development benchmarks and selection

- `artifacts/runs/classical_nested_v2/`
- `artifacts/runs/gnn_nested_v2/`
- `artifacts/runs/development_selection_v2/`
- Selection SHA-256:
  `ac5b84f2f37cb71f9503bfcc4abdad8674f54b157da1e6036fc20148c8e26519`

Model selection used repeated scaffold-disjoint development OOF predictions only.
The selected model was `ridge_combined` with mean repeat RMSE 0.741813.

### One-time temporal evaluation

- `artifacts/runs/temporal_evaluation_v2/`
- Evaluation SHA-256:
  `6a91f9fd6d221b3b1be8315b92d863e708c44894649312cbf70beae04c1b5c31`
- Primary temporal RMSE: 0.972440
- Primary temporal MAE: 0.750101
- Primary temporal R2: 0.469763

The directory contains `RUN_COMPLETE`, all output hashes, the temporal prediction
table, diagnostics, final training manifest, and the serialized Ridge pipeline.
The runner verifies this completed result without rereading temporal labels.

### Post hoc temporal reliability audit

- Source: `artifacts/runs/temporal_evaluation_v2/temporal_predictions.csv`
- Generator: `scripts/08_build_manuscript_assets.py`
- Machine-readable output: `manuscript_v2/generated/manuscript_results.json`
- Human-readable output: the annual table and temporal-evaluation figure under
  `manuscript_v2/`

This audit decomposes between- and within-year behavior and evaluates annual
error and rank association. It is labeled `post_hoc_descriptive`, was computed
only after the sealed primary evaluation, and must not be represented as a
prespecified selection or confirmatory analysis.

### Post hoc robustness and source-dependence audit

- `artifacts/audits/temporal_robustness_v2/`
- Generator: `scripts/10_build_posthoc_robustness_audit.py`
- Audit SHA-256:
  `8aaaa1997993e78f4a6e6b0fcf076702295770d53716cb7a89a9eacdf7f94a5b`

This hash-linked audit consumes the immutable selected-model prediction table
without refitting or changing any prediction. It records strict-versus-broad
chronology checks, scaffold/assay-source/document-source bootstrap sensitivities,
uncertainty for the 2019-to-2024 contrast, source-centered ordering, and the
nearest-neighbor potency-discordance analysis. Every result is explicitly post
hoc and cannot be used to revise model selection or the frozen primary result.

### Post hoc assay-provenance and GNN-duration audits

- Assay source: `data/curated/chembl37_pim1_v2/strict_measurement_events.csv`
- Split sources: the immutable `development.csv` and `temporal_test_sealed.csv`
- GNN source: all JSON/CSV pairs under
  `artifacts/runs/gnn_nested_v2/checkpoints/`
- Generator and machine-readable output: the same manuscript builder and results
  JSON listed above
- Human-readable assay output:
  `manuscript_v2/generated/table_assay_context.tex`

The assay audit is labeled `post_hoc_descriptive`; it counts archived metadata,
partition provenance, and all within-document identity bridges across explicit
ATP conditions. It does not estimate assay effects or modify potency labels.
The GNN audit is labeled `realized_checkpoint_audit`; it verifies training
durations and prediction hashes but does not retrain or expand the candidate set.

### Post hoc expanded graph benchmark

- Search contract: `configs/gnn_search_v1.yaml`
- Runner: `scripts/11_run_gnn_tuned_nested.py`
- Implementation: `src/pim1_qsar/benchmark/gnn_tuned.py`
- Configurable GINE: `src/pim1_qsar/models/gine_tuned.py`
- Artifacts: `artifacts/runs/gnn_tuned_posthoc_v1/`
- Manuscript table: `manuscript_v2/generated/table_tuned_gnn.tex`
- Exact supplementary search-space table:
  `manuscript_v2/generated/table_tuned_gnn_search_space.tex`

This analysis reuses only the immutable development table and outer scaffold
assignments. It performs nested hyperparameter search, initialization reranking,
and outer seed ensembling for task-trained GINE, task-trained Chemprop D-MPNN,
and a frozen CheMeleon representation. The run manifest hashes the search
contract, code, GNN environment, pretrained checkpoint, split inputs, reference
Ridge OOF predictions, and all summary outputs. It is labeled
`post_hoc_development_only` because the temporal outcomes had already been
inspected before the expanded search was designed; it cannot alter the frozen
model selection or temporal result.

Fields prefixed `conformal_` in the sealed temporal schema are historical
internal names for the descriptive OOF-residual reference band. They do not
denote a formally valid conformal predictor.

### Manuscript

- `manuscript_v2/main.tex`
- `manuscript_v2/main.pdf`
- `manuscript_v2/references.bib`
- `manuscript_v2/generated/manuscript_results.json`
- `manuscript_v2/generated/asset_manifest.json`
- `manuscript_v2/manuscript_manifest.json`
- `manuscript_v2/MANUSCRIPT_COMPLETE`
- `manuscript_v2/figures/`

All numerical manuscript macros and tables are generated by
`scripts/08_build_manuscript_assets.py` from the completed artifacts. The final
PDF and sources are verified and sealed by `scripts/09_seal_manuscript.py`.

## Legacy: exploratory/development history only

- Complete snapshot: `archive/legacy_v1/`
- Paper: `archive/legacy_v1/paper.tex` and `archive/legacy_v1/paper.pdf`
- Numbered scripts: `archive/legacy_v1/1.*` through `7.*`
- Results and models: `archive/legacy_v1/results/CHEMBL2147/` and
  `archive/legacy_v1/models/CHEMBL2147/`
- Older 3,110-compound CSV files: `archive/legacy_v1/data/`
- Runners: `archive/legacy_v1/run_pipeline.sh`,
  `archive/legacy_v1/run_vm_evaluation.sh`
- Integrity inventory: `archive/legacy_v1/ARCHIVE_MANIFEST.json`, verified by
  `scripts/12_manage_archive_manifest.py`

Legacy results used an older broad cohort, partial nesting, and a fixed reused
scaffold holdout. They must be described only as prior exploratory work and are
not a sensitivity analysis of v2.

Non-evidentiary manuscript-rendering scratch and machine-local cache state
(`archive/workspaces/` and `archive/local_state/`) are not part of the public
tree; they are gitignored and must not be reintroduced.

## Submission checklist

- [ ] Add a public repository commit/tag containing the canonical v2 files.
- [ ] Deposit an archival snapshot and insert its DOI in the manuscript.
- [ ] Replace remaining placeholders: ORCID, named AI/LLM tools, public GitHub URL, and Zenodo DOI.
- [ ] Confirm author name, affiliation, correspondence, funding, and acknowledgments.
- [ ] Choose a journal and apply its article template and reporting requirements.
- [ ] Preserve the completed temporal directory and all SHA-256 manifests.
- [ ] Preserve the completed post hoc audit directory and its input hashes.
- [ ] Do not describe the retrospective temporal split as prospective validation.
- [ ] Do not claim selectivity, cellular efficacy, ADMET, or therapeutic activity.
