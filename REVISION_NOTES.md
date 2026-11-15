# Revision Notes — pre-submission patch pass

## 2026-11-15 public-release packaging

Packaging and manuscript metadata only. Canonical scientific artifacts, curated
tables, split locks, run outputs, the temporal evaluation directory, and audit
directories were not modified, and no reported numbers were changed.

- Added manuscript affiliation, correspondence email, an ORCID placeholder,
  a funding statement, hyperref PDF metadata, a tightened AI-tool disclosure,
  and public GitHub/Zenodo availability placeholders.
- Added `DATA_LICENSE.md` and `NOTICE` so ChEMBL-derived files are licensed
  CC BY-SA 3.0 with required attribution, while code remains MIT. Chemprop and
  the hash-pinned CheMeleon checkpoint are recorded as third-party MIT
  components that are not redistributed here.
- Removed non-evidentiary `archive/workspaces/` and `archive/local_state/`
  from the public tree and gitignored them. Scientific files under
  `archive/legacy_v1/` were not modified. The legacy SHA-256 inventory was
  rewritten to the 186 git-tracked files; gitignored LaTeX aux, `*.log`, and
  `.DS_Store` entries were never present in a clone and are now skipped by
  the verifier.
- Corrected documentation that still described the sealed PDF as 21 pages; the
  sealed manuscript is 25 pages.

## 2026-07-20 project structure consolidation

The canonical v2 study and the earlier broad-cohort workflow are now physically
separated without deleting historical material.

- Moved the complete pre-v2 workflow into `archive/legacy_v1/`, preserving its
  scripts, support modules, inputs, tests, paper, models, results, plots, and
  logs in a self-contained relative layout.
- Moved non-canonical manuscript-rendering scratch into `archive/workspaces/`
  and retained machine caches/session files under `archive/local_state/`
  (later removed from the public tree; see the 2026-11-15 note).
- Added `docs/PROJECT_STRUCTURE.md`, explicit archive documentation, a
  192-file SHA-256 archive inventory, and a verifier at
  `scripts/12_manage_archive_manifest.py`.
- Updated the artifact index, README, CI workflow, and historical path
  references to reflect the clean evidentiary boundary.
- Added structural regression tests. Final validation passed 96 canonical v2
  tests with three optional-dependency skips and 33 preserved v1 tests. The
  sealed manuscript hash and all completed-run artifacts remained unchanged.

## 2026-07-13 expanded graph-model benchmark

The original graph arm was a controlled representation ablation, not a
hyperparameter-matched benchmark. A new versioned development-only analysis now
gives learned graph models a substantially stronger comparison without
retroactively reusing the already-opened temporal holdout for selection.

- Added a 24-candidate search (one anchored baseline plus 23 deterministic
  random draws) per family across optimizer, depth, width, head, dropout,
  pooling, normalization, batch size, descriptor fusion, featurization, and
  backend-specific residual/initialization choices. The same family-specific
  candidate set was evaluated independently within every outer fold.
- Added top-three inner-CV reranking across three initialization seeds and
  three-seed outer ensembles, retaining seed-specific predictions.
- Corrected the prior optimizer asymmetry by implementing explicit AdamW weight
  decay for Chemprop. Validation targets and descriptors now use training-fold
  scalers; Chemprop early stopping monitors validation RMSE in scaled space,
  with the improvement threshold and reported fold score exactly converted to
  raw pIC50 RMSE.
- Added a frozen 2,048-dimensional CheMeleon pretrained D-MPNN representation
  with a hash-pinned Zenodo checkpoint and explicit structural-exposure caveat.
- Added complete inner-trial, selected-configuration, seed-prediction, ensemble,
  repeat-metric, paired scaffold-bootstrap, environment, code, and checkpoint
  lineage.
- Kept the sealed v2 selection and temporal predictions unchanged; the expanded
  graph result is labeled `post_hoc_development_only` throughout the code,
  artifacts, paper, and documentation.
- Completed and hash-sealed all 1,350 trial objects, 4,050 inner-fold fits, and
  45 outer selections. Frozen CheMeleon was the strongest graph family (RMSE
  0.754 +/- 0.011), followed by GINE (0.760 +/- 0.012) and task-trained D-MPNN
  (0.779 +/- 0.011). The best graph-family paired difference versus locked
  Ridge was +0.002 (95% scaffold-cluster CI -0.038 to +0.046), so the stronger
  comparison did not establish a graph-model advantage.
- Final validation passed 96 v2 tests locally (three optional-dependency skips,
  including the later project-structure guards)
  and 21 targeted GNN tests on the VM (one expected skip), including independent
  recomputation of tuned ensembles, repeat summaries, and Ridge differences.

## 2026-07-12 deep-dive revision — canonical v2

The release-pinned v2 benchmark and manuscript received a separate scientific,
statistical, code, provenance, literature, and rendered-PDF audit. The frozen
model selection and one-time temporal predictions were not changed.

- Added `scripts/10_build_posthoc_robustness_audit.py` and reusable audit helpers
  with code, runtime, input, output, seed, and completion hashes.
- Quantified uncertainty for the 2019-to-2024 RMSE change and added assay-source
  and document-source-set dependence sensitivities.
- Added within-document and within-assay centered ranking analyses and an
  explanatory nearest-neighbor potency-discordance analysis.
- Corrected ``unseen/first seen'' wording after finding four temporal identities
  with earlier broad but non-strict records; excluding them leaves the primary
  conclusion unchanged.
- Added current primary literature on inter-assay IC50 variability and activity
  cliffs, corrected ``structure-based'' to ``ligand-based,'' and clarified that
  the temporal seal is an auditable software control rather than cryptographic
  inaccessibility.
- Added three audit contract tests, repaired the canonical artifact index,
  removed stale machine-specific bytecode, regenerated all manuscript assets,
  and visually verified the clean 25-page PDF.

Validation after this pass: 114 passed and 2 optional-dependency skips in the
complete classical test suite; 12 passed and 1 expected skip in the targeted GNN
suite. The legacy notes below document the earlier pre-v2 pipeline history.

Changes applied in response to the internal peer review. Grouped by whether they are
**safe** (no pipeline re-run needed; reported numbers unaffected) or **deferred**
(would change results and require re-running the pipeline to re-report).

## Applied — legacy paper (`archive/legacy_v1/paper.tex`)

1. **§3.3 significance contradiction fixed.** Removed "several comparisons achieved p < 0.05"
   (contradicted Table 2, where every p ≥ 0.063 / Sig. = No). Replaced with the correct
   statement: at n = 5 paired folds the two-sided Wilcoxon floor is 2/2⁵ = 0.0625, so no
   comparison can reach p < 0.05 — the top models are statistically indistinguishable.
2. **§3.1 pIC50 range corrected.** "≈4.0 to 10.0" → "≈1.25 to 11.22 (median 7.61)", with an
   explicit note on the tail (141 compounds, 4.5%, pIC50 > 10; 4 compounds < 4) as likely
   ChEMBL data-quality/heterogeneity artifacts.
3. **§3.6 interpretability model corrected.** "VotingEnsemble" → "best-performing model
   (StackingEnsemble)", matching the code's best-by-R² selection in `7.Model_Interpretability.py`.
4. **§4.1 overstatement softened.** "substantially exceeds these benchmarks" → an
   in-range framing with an explicit apples-to-oranges caveat vs. Martin et al. (per-assay
   medians under prospective splitting vs. our single dense target under scaffold split).
5. **§2.4 preprocessing description corrected.** Now states the real routing: PCA for
   linear/distance models, raw 2048-bit fingerprints for the standalone tree models.
6. **§4.3 rewritten** ("PCA Limitation" → "Representation and the Tree/Linear Gap").
   The old text blamed PCA for the trees losing, but standalone trees received *raw*
   fingerprints and still underperformed (RF 0.607 vs. Ridge 0.680). New text says this,
   and discloses the ensemble caveat (Voting/Stacking base-learner trees run on PCA features).

## Applied — code (`5.Comparing_Regressors.py`)

7. **Y-randomization seeded.** `np.random.permutation` → seeded `RandomState(42).permutation`,
   so the Y-rand result is reproducible (everything else in the file was already seeded).
   NOTE: the reported value (test R² = −0.005) should be re-generated on the next run; it
   will remain near-zero but the exact figure changes with the seed.
8. **Transformer `.pkl` overwrite bug fixed.** The imputer/PCA/scaler/mask were dumped inside
   the per-model loop; the last model (DecisionTree, `skip_pca=True`) saved
   `morgan_transformer.pkl` as `None`, which scripts 6 & 7 then loaded. Now guarded to persist
   only from `skip_pca=False` models, so the PCA representation used by the best model / Ridge
   interpretability is the one saved. Does not affect Table 1/2 numbers.
9. **Stale comments fixed.** `n_iter=50` comment → 20 (code always used 20); `"5255 samples"`
   → sample-size-agnostic wording.

## Applied — data (all committed CSVs + `molecule.smi` + `target_information.csv`)

10. **Removed a second target (PAK4 / CHEMBL4482).** The committed data files contained PAK4
    rows that the archived `1.Download` script never fetches, so the legacy
    `archive/legacy_v1/run_pipeline.sh` could not
    regenerate them (a reproducibility/provenance defect, though the reported PIM1 results were
    unaffected — `5.py` filters on `--target-id CHEMBL2147` and errors on mixed targets).
    All files are now PIM1-only and internally consistent with the paper's N = 3,110:
    - `..._3class_pIC50_morgan_fp.csv` (modeling): 3,271 → **3,110** rows (3,110 unique molecules)
    - `bioactivity_data_raw.csv`, `bioactivity_data_preprocessed.csv`,
      `..._2class_lipinski.csv`: 5,055 → **4,852** rows
    - `molecule.smi`: 4,391 → **3,122** lines
    - `target_information.csv`: PAK4 row removed
    Pristine originals backed up under the session scratchpad (`data_backup/`).

## Applied — pipeline re-run (regenerated the missing results/models/plots)

- **matplotlib 3.11 compat fix** (`6.Applicability_Domain.py:471`): `boxplot(labels=...)` →
  `tick_labels=` (the `labels` kwarg was removed in matplotlib 3.11; step 6 crashed on it).
- **Ran steps 2, 4, 5, 6, 7** on the cleaned v33 data in a fresh venv (numpy 2.5.0, pandas 3.0.3,
  scikit-learn 1.9.0, rdkit 2026.3.3, matplotlib 3.11). All steps exit 0. Regenerated the
  previously-absent `archive/legacy_v1/results/CHEMBL2147/`,
  `archive/legacy_v1/models/CHEMBL2147/`, and
  `archive/legacy_v1/plots/CHEMBL2147/`.
  Skipped step 1 (v33 lock) and step 3 (removed rdkit API; descriptors already present).
- **Verified fix #8 end-to-end:** `morgan_transformer.pkl` is now a fitted PCA (627 components),
  not `None`.

### Reproduction vs. the manuscript
- **Headline reproduces cleanly:** StackingEnsemble test R² 0.701 → **0.702**; bootstrap CI
  0.701 [0.643, 0.746] → 0.702 [0.643, 0.747]; Y-randomization −0.005 → **−0.004** (now seeded);
  AD in-domain R² 0.713 > overall 0.702 > outside 0.189. All Wilcoxon p ≥ 0.0625 — validates the
  corrected §3.3.
- **Tree models are unstable across library versions** (RandomizedSearch sampling depends on the
  numpy/scipy RNG stream): RandomForest **0.607 → 0.694** (now 2nd-best, above Ridge),
  DecisionTree **0.367 → 0.571**, GradientBoosting 0.612 → 0.637. Linear / instance / ensemble
  models reproduce to within ±0.001.
- **Historical decision:** the fresh v1 run was adopted into
  `archive/legacy_v1/paper.tex`, which was updated to the regenerated
  numbers throughout (verified by a 3-agent cross-check against the CSVs — 80 checks, 0 discrepancies):
  - **Table 1** reordered by the new holdout R² (RandomForest is now 2nd at 0.694; DecisionTree
    0.367→0.571); all CV means/stds refreshed.
  - **Table 2** regenerated (StackingEnsemble reference; all comparisons still non-significant).
  - Headline numbers updated everywhere: 0.701→**0.702**, RMSE 0.715→**0.714**, bootstrap CIs,
    Y-rand −0.005→**−0.004** (abstract, §3.2, §3.3, §3.4, §4.1, §4.2, Conclusions).
  - **§3.5 (AD)** made concrete: 93.8% in-domain; in-domain R² 0.713 (RMSE 0.688) vs. outside
    0.189 (RMSE 1.031).
  - **§3.6 (interpretability)** now reports real descriptors (resolves reviewer §4.5): FractionCSP3
    leads (permutation importance and Ridge coefficient +0.19), then MolWt (+0.06), TPSA (+0.03),
    Chi0v (−0.06); largest coefficients overall are Morgan-derived PCA components.
  - **§4.3** reframed ("Model Representation"): the "linear beats trees" narrative is removed
    (RandomForest on raw fingerprints is now competitive/tied); the ensemble-PCA caveat is kept as
    a concrete improvement path. Conclusions note the best model is statistically indistinguishable
    from RandomForest and the linear models.
  - Because the tree models proved sensitive to library-version-driven RandomizedSearch sampling,
    the multi-seed robustness analysis (reviewer §4.3) remains the recommended next step.

## Deferred — need a pipeline re-run to re-report (NOT silently changed)

- **Ensemble representation** — give Voting/Stacking tree base-learners raw fingerprints via
  per-estimator pipelines (§4.3 now discloses the current caveat). Changes the reported
  ensemble numbers; requires a re-run + table update.
- **Outlier trimming** — the 141 compounds with pIC50 > 10 (IC50 < 100 pM) are left in; the
  text now flags them. Decide whether to trim and re-report whether metrics move.
- **Multi-seed scaffold splits** (reviewer §4.3) — `random_state=42` is hardcoded in ~36 sites
  across 6 files; thread a single seed variable, run 3–5 seeds, report mean ± std.
- **Temporal (deposition-year) external split** (reviewer §4.4) — `document_year` exists in the
  raw ChEMBL pull (2005–2024) but is dropped at step 1; re-join it onto the 3,110-row set by
  `molecule_chembl_id` (years are skewed, ~40% in 2016).
- **GNN / Chemprop baseline** (reviewer §4.8) — net-new dependency + graph featurization.

## Also recommended (authorial, not applied)

- De-emphasize PDAC framing in the Introduction and the `pancreatic_cancer_*` file names
  (the title is already PIM1-only; §4.4 biological framing already hedges honestly).
