---
name: biodd-pim1-qsar-project
description: "State of Tair Narynov's PIM1 kinase QSAR paper/codebase (biodd_rerun) — audited, patched, re-run, fresh numbers adopted."
metadata: 
  node_type: memory
  type: project
  originSessionId: 3843ac6b-d40f-41b4-852b-b4cbd577920d
---

**Project:** "Scaffold-Aware QSAR Modeling of PIM1 Kinase Inhibitors" (author Tair Narynov) — a
cheminformatics ML pipeline + LaTeX manuscript being prepped for a mid-tier cheminformatics journal.
Predicts pIC50 vs PIM1 kinase (ChEMBL `CHEMBL2147`) from Morgan/ECFP6 fingerprints + 23 RDKit
physicochemical descriptors; Bemis–Murcko scaffold split; nested 5×3 CV; 11 regressors incl.
Voting/Stacking ensembles. Lives at `/home/bmddjul1/extracted_rerun/` (from `biodd_rerun.zip`).

**What I did (this session):** audited an internal peer review against the code (mostly correct);
applied patches (fixed a §3.3 significance self-contradiction, corrected pIC50 range, softened the
Martin "substantially exceeds" claim, seeded Y-randomization, fixed a transformer-`.pkl` overwrite
bug, **removed a stray second kinase PAK4/CHEMBL4482 that contaminated the committed CSVs**); built a
Python env from scratch; **re-ran steps 2,4,5,6,7** (skipped 1=v33 lock, 3=removed rdkit API);
**adopted the fresh run into `paper.tex`** (verified: 80 agent checks, 0 discrepancies).

**Current headline:** StackingEnsemble test R²=0.702, RMSE=0.714; RandomForest now 2nd (0.694, up
from 0.607 in the original — tree models shifted a lot due to newer libs changing RandomizedSearch
sampling); all Wilcoxon p ≥ 0.0625 (nothing significant); Y-rand −0.004; AD in-domain 0.713 vs
outside 0.189; top descriptor FractionCSP3. The "linear beats trees" narrative was retired (§4.3
reframed "Model Representation").

**Key gotchas:** VM had no pip/ensurepip → bootstrap via `python3 -m venv --without-pip` + get-pip.py;
`requirements.txt` pins `rdkit-pypi==2025.9.6` which does not exist (use `rdkit`); rdkit 2026 removed
`AllChem.GetMorganFingerprintAsBitVect` (step 3 crashes); matplotlib 3.11 removed `boxplot(labels=)`
(use `tick_labels=`); `biodd_env/` and `models/` (199 MB pickles) are big/regenerable.

**Full detail:** see `extracted_rerun/REVISION_NOTES.md` (change log) and
`extracted_rerun/PROJECT_UNDERSTANDING.md` (snapshot). **Top deferred item:** multi-seed robustness
(reviewer §4.3) — most valuable next step given the tree instability. Also: temporal split
(`document_year` in raw), ensemble per-estimator representation, outlier trimming (141 cmpds pIC50>10),
GNN baseline. See [[biodd-pim1-qsar-project]].
