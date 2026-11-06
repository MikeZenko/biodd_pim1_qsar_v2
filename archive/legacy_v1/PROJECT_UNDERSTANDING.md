# PROJECT_UNDERSTANDING.md — Claude's snapshot of this project

_Written by Claude (Opus 4.8) at the end of a working session. This is my understanding of the
project in its current state, so it can be picked up again later. Full change log is in
`REVISION_NOTES.md`._

## 1. What this project is

**Scaffold-Aware QSAR Modeling of PIM1 Kinase Inhibitors Using Ensemble Machine Learning** — a
cheminformatics / ML pipeline plus a LaTeX manuscript (author: **Tair Narynov**), being prepared
for submission to a mid-tier cheminformatics journal.

- **Goal:** predict bioactivity (pIC50) of small molecules against **PIM1 kinase**
  (ChEMBL target `CHEMBL2147`) from molecular structure — a QSAR regression model to prioritize
  candidate inhibitors.
- **Data:** ChEMBL v33, IC50 (`=` relations only), median-deduplicated → **3,110 unique PIM1
  compounds**. Features = Morgan/ECFP6 fingerprints (radius 3, 2048 bits) + 23 physicochemical
  descriptors (RDKit).
- **Method:** Bemis–Murcko **scaffold-based** 80/20 split; nested CV (5 outer × 3 inner, GroupKFold);
  11 regression algorithms incl. Voting/Stacking ensembles; leakage-safe preprocessing fitted inside
  each fold; validation battery (Y-randomization, k-NN applicability domain, bootstrap CIs,
  Wilcoxon + Nadeau–Bengio + Cohen's d); aligned to the 5 OECD QSAR principles.
- **Headline result (current):** StackingEnsemble **test R² = 0.702, RMSE = 0.714** pIC50 units,
  statistically indistinguishable from RandomForest and the regularized linear models.

## 2. Current state — what happened this session

The work went: **peer-review audit → patch fixes → build env → re-run pipeline → adopt the fresh run
into the paper.** Sequence:

1. **Audited an internal peer review against the code** (it was largely correct). Confirmed the
   leakage controls, scaffold integrity, and validation battery are real. Found issues the review
   missed and corrected two it got wrong. See `REVISION_NOTES.md`.
2. **Applied safe patches** (paper text + code + data):
   - Paper: fixed the §3.3 significance self-contradiction; corrected the pIC50 range; softened the
     "substantially exceeds" Martin comparison; fixed the §2.4/§4.3 PCA-routing description.
   - Code (`5.Comparing_Regressors.py`): **seeded the Y-randomization**; **fixed a transformer
     `.pkl` overwrite bug** (tree models were saving `morgan_transformer.pkl` as `None`); fixed
     stale comments.
   - Data: **removed a stray second kinase (PAK4 / CHEMBL4482)** that had contaminated the committed
     CSVs (the download script only fetches PIM1, so the committed data couldn't be regenerated).
     All data files are now PIM1-only and consistent with N = 3,110.
3. **Built a Python environment from scratch** (the VM had no pip/numpy/rdkit): bootstrapped pip via
   `get-pip.py` into `biodd_env/`, installed numpy 2.5, pandas 3.0.3, scipy 1.18, scikit-learn 1.9,
   rdkit 2026.3.3, matplotlib 3.11.
4. **Re-ran the modeling pipeline** (`run_local.sh`) — steps 2, 4, 5, 6, 7 — regenerating the
   previously-missing `results/`, `models/`, `plots/`. Fixed a matplotlib-3.11 `boxplot(labels=)` →
   `tick_labels=` crash in step 6.
5. **Adopted the fresh run as canonical**: rewrote `paper.tex` Table 1, Table 2, bootstrap CIs,
   Y-rand, abstract/§3.2/§3.3/§3.4/§4.1/§4.2/Conclusions, made §3.5 (AD) and §3.6 (interpretability)
   concrete, and reframed §4.3. Verified by a 3-agent cross-check against the CSVs (80 checks,
   0 discrepancies) plus a stale-number grep.

### Key narrative change from the re-run
The tree models moved a lot vs. the original paper (RandomForest **0.607 → 0.694**, DecisionTree
0.367 → 0.571) — purely from newer numpy/scipy changing the `RandomizedSearchCV` sampling. Linear /
instance / ensemble models reproduced to ±0.001 (bootstrap CI and Y-rand near-identical). So:
- RandomForest is now the **2nd-best** model; the old "linear beats trees" story is retired.
- The instability **empirically motivates the reviewer's multi-seed recommendation** (top deferred item).

## 3. Current headline numbers (fresh run, in `results/CHEMBL2147/`)

- Test R²: StackingEnsemble 0.702, RandomForest 0.694, ElasticNet 0.681, Ridge 0.680,
  VotingEnsemble 0.676 (top-5 band 0.676–0.702), … AdaBoost 0.422, DecisionTree 0.571.
- Bootstrap 95% CI (Stacking): R² 0.702 [0.643, 0.747]; RMSE 0.714 [0.658, 0.770]; MAE 0.542 [0.503, 0.583].
- Y-randomization: mean test R² −0.004 (PASSED, now seeded/reproducible).
- Statistical comparison: all Wilcoxon p ≥ 0.0625 (n=5 floor) → nothing significant.
- Applicability domain: 93.8% in-domain; in-domain R² 0.713 (RMSE 0.688) vs. outside 0.189 (1.031).
- Interpretability: top physchem descriptor = **FractionCSP3** (permutation importance and Ridge
  coefficient +0.19), then MolWt (+0.06), TPSA (+0.03), Chi0v (−0.06); largest Ridge coefficients
  overall are Morgan-derived PCA components.

## 4. File / directory map

```
1..7.*.py                     Pipeline steps (1 download, 2 EDA, 3 descriptors, 4 RF baseline,
                              5 model comparison [core], 6 applicability domain, 7 interpretability)
preprocessing_utils.py        Leakage-safe preprocessing (imputer/PCA/scaler/MI-selector, skip_pca)
scaffold_utils.py             Bemis–Murcko scaffold split (GroupShuffleSplit / GroupKFold)
paper.tex                     Manuscript — UPDATED to the fresh run
REVISION_NOTES.md             Full change log (patches + reproduction + deferred items)
PROJECT_UNDERSTANDING.md      This file
run_local.sh                  Driver that ran steps 2,4,5,6,7 on the locked data (skips 1 & 3)
data/                         Cleaned, PIM1-only inputs (3,110-compound modeling CSV, etc.)
results/CHEMBL2147/           Regenerated tables, stats, CIs, importances, per-fold scores
models/CHEMBL2147/            Regenerated model pickles (sklearn 1.9 — version-locked, ~199 MB)
plots/, plots/CHEMBL2147/     Regenerated figures (PNG + PDF)
pipeline_run.log              Log of the full re-run
requirements.txt              Original pins (NOTE: rdkit-pypi==2025.9.6 does not exist — use `rdkit`)
```

## 5. Environment / reproduction gotchas (important)

- The VM had **no pip and no ensurepip**. To rebuild the env: `python3 -m venv --without-pip biodd_env`,
  fetch `get-pip.py` from bootstrap.pypa.io, run `biodd_env/bin/python get-pip.py`, then
  `biodd_env/bin/pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib tqdm requests pillow rdkit`.
  (The venv `biodd_env/` was **excluded from the zip** — it is machine-specific; rebuild it.)
- `requirements.txt` pins are partly unbuildable: **`rdkit-pypi==2025.9.6` does not exist** — install
  `rdkit` instead. Installed versions differ from the pins (newer), which is why the tree-model
  numbers shifted.
- **rdkit 2026** removed `AllChem.GetMorganFingerprintAsBitVect` (used in `3.Descriptor…py:97`), so
  **step 3 will crash** on current rdkit. This is why the re-run skipped step 3 and reused the
  committed descriptor matrix. To run step 3, port it to `rdkit.Chem.rdFingerprintGenerator`.
- **Step 1 (download) was skipped on purpose** — a live pull hits current ChEMBL (v36+), which would
  break the v33 reproducibility lock and change N off 3,110.
- matplotlib 3.11 removed `boxplot(labels=)` → use `tick_labels=` (already fixed in step 6).

## 6. Deferred / recommended next steps (not done)

1. **Multi-seed robustness** (highest value) — thread a single seed through the ~36 hardcoded
   `random_state=42` sites, run 3–5 seeds, report mean ± std. Directly answers the reviewer and the
   observed tree instability.
2. **Temporal (deposition-year) external split** — `document_year` exists in the raw ChEMBL pull
   (2005–2024) but is dropped at step 1; re-join by `molecule_chembl_id`.
3. **Ensemble representation refactor** — give Voting/Stacking tree base-learners raw fingerprints
   via per-estimator pipelines (§4.3 currently discloses this as a caveat).
4. **Outlier decision** — 141 compounds have pIC50 > 10 (IC50 < 100 pM); text flags them; decide on trimming.
5. **GNN / Chemprop baseline** (venue-dependent).
6. Housekeeping: `models/` has redundant pickles (step-4 vs step-5 naming: `ensemble_model.pkl` 68 MB,
   `random_forest_model.pkl` vs `randomforestregressor_tuned.pkl`, `*_ensemble.pkl` vs `*ensemble.pkl`).

## 7. Repo / provenance

- GitHub: github.com/MikeZenko/Scaffold-Aware-QSAR-Modeling-PIM1 (per README).
- The paper's numbers are now from the **fresh local run** (sklearn 1.9), not the author's original
  pinned environment. If the author prefers their original environment's numbers, revert `paper.tex`
  to the pre-adoption values (recorded implicitly in git / the original zip) — but the fresh numbers
  are what reproduce today.
