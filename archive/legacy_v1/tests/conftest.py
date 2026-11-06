import numpy as np
import pandas as pd
import pytest


BENZENE_DERIVATIVES = [
    "c1ccccc1",        # benzene
    "c1ccc(O)cc1",     # phenol
    "c1ccc(N)cc1",     # aniline
    "c1ccc(F)cc1",     # fluorobenzene
]

NAPHTHALENE_DERIVATIVES = [
    "c1ccc2ccccc2c1",         # naphthalene
    "c1ccc2cc(O)ccc2c1",      # 2-naphthol
    "c1ccc2cc(N)ccc2c1",      # 2-naphthylamine
]

PYRIDINE_DERIVATIVES = [
    "c1ccncc1",        # pyridine
    "c1ccnc(N)c1",     # 2-aminopyridine
    "c1ccnc(O)c1",     # 2-hydroxypyridine
]

BIPHENYL_DERIVATIVES = [
    "c1ccc(-c2ccccc2)cc1",         # biphenyl
    "c1ccc(-c2ccc(O)cc2)cc1",      # 4-hydroxybiphenyl
    "c1ccc(-c2ccc(N)cc2)cc1",      # 4-aminobiphenyl
    "c1ccc(-c2ccc(F)cc2)cc1",      # 4-fluorobiphenyl
]

CYCLOHEXANE_DERIVATIVES = [
    "C1CCCCC1",        # cyclohexane
    "C1CCCC(O)C1",     # cyclohexanol
    "C1CCCC(N)C1",     # cyclohexylamine
]


@pytest.fixture
def diverse_smiles():
    """20 SMILES spanning 5 distinct Bemis-Murcko scaffolds."""
    return (
        BENZENE_DERIVATIVES
        + NAPHTHALENE_DERIVATIVES
        + PYRIDINE_DERIVATIVES
        + BIPHENYL_DERIVATIVES
        + CYCLOHEXANE_DERIVATIVES
    )


@pytest.fixture
def diverse_features(diverse_smiles):
    """Synthetic feature matrix aligned to diverse_smiles."""
    rng = np.random.RandomState(42)
    return rng.rand(len(diverse_smiles), 10)


@pytest.fixture
def diverse_targets(diverse_smiles):
    """Synthetic pIC50 values aligned to diverse_smiles."""
    rng = np.random.RandomState(42)
    return rng.uniform(4.0, 9.0, size=len(diverse_smiles))
