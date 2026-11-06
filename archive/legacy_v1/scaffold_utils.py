"""
Scaffold-based data splitting utilities for QSAR modeling.

This module provides functions for extracting molecular scaffolds and
performing scaffold-based train/test splitting to ensure realistic
performance estimates.

Reference: Bemis-Murcko scaffold analysis prevents data leakage from
structurally similar molecules appearing in both train and test sets.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from collections import defaultdict

logger = logging.getLogger(__name__)

_scaffold_fallback_count = 0


def get_scaffold_fallback_count():
    """Return the number of SMILES that fell back to singleton scaffolds."""
    return _scaffold_fallback_count


def reset_scaffold_fallback_count():
    """Reset the fallback counter (useful for testing)."""
    global _scaffold_fallback_count
    _scaffold_fallback_count = 0


def get_scaffold(smiles, include_chirality=False):
    """
    Extract Bemis-Murcko scaffold from a SMILES string.
    
    Parameters:
    -----------
    smiles : str
        SMILES string of the molecule
    include_chirality : bool
        Whether to include stereochemistry in scaffold
        
    Returns:
    --------
    str : Scaffold SMILES or original SMILES if extraction fails
    """
    global _scaffold_fallback_count
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            _scaffold_fallback_count += 1
            msg = f"Could not parse SMILES: {smiles!r}, using as singleton scaffold"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            return smiles
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, 
            includeChirality=include_chirality
        )
        if not scaffold:
            _scaffold_fallback_count += 1
            msg = f"Empty scaffold for SMILES: {smiles!r}, using as singleton scaffold"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            return smiles
        return scaffold
    except Exception:
        _scaffold_fallback_count += 1
        msg = f"Scaffold extraction failed for SMILES: {smiles!r}, using as singleton scaffold"
        logger.warning(msg)
        warnings.warn(msg, stacklevel=2)
        return smiles


def get_scaffold_groups(smiles_list):
    """
    Convert SMILES list to scaffold group indices.
    
    Parameters:
    -----------
    smiles_list : list or array-like
        List of SMILES strings
        
    Returns:
    --------
    numpy.ndarray : Array of integer group labels
    dict : Mapping of scaffold SMILES to group index
    """
    count_before = _scaffold_fallback_count
    scaffold_to_idx = {}
    groups = []
    
    for smiles in smiles_list:
        scaffold = get_scaffold(smiles)
        if scaffold not in scaffold_to_idx:
            scaffold_to_idx[scaffold] = len(scaffold_to_idx)
        groups.append(scaffold_to_idx[scaffold])
    
    fallbacks = _scaffold_fallback_count - count_before
    if fallbacks > 0:
        msg = (f"{fallbacks}/{len(smiles_list)} molecules fell back to "
               f"singleton scaffolds (unparseable or empty scaffold)")
        logger.warning(msg)
        print(f"  WARNING: {msg}")
    
    return np.array(groups), scaffold_to_idx


def scaffold_split(X, y, smiles, test_size=0.2, random_state=42):
    """
    Perform scaffold-based train/test split.
    
    Ensures that molecules with the same scaffold are either all in
    training or all in test set, preventing data leakage.
    
    Parameters:
    -----------
    X : DataFrame or array-like
        Feature matrix
    y : Series or array-like
        Target values
    smiles : list or array-like
        SMILES strings corresponding to each sample
    test_size : float
        Fraction of scaffold groups for test set (default: 0.2). The resulting
        compound fraction can differ because scaffold groups have unequal size.
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test, scaffold_groups
    """
    # Get scaffold groups
    scaffold_groups, scaffold_mapping = get_scaffold_groups(smiles)
    
    # Use GroupShuffleSplit for scaffold-based splitting
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=scaffold_groups))
    
    # Convert to appropriate types
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
    
    if isinstance(y, pd.Series):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]
    
    # Log scaffold statistics
    train_scaffolds = set(scaffold_groups[train_idx])
    test_scaffolds = set(scaffold_groups[test_idx])
    overlap = train_scaffolds & test_scaffolds
    
    print(f"\nScaffold-based split statistics:")
    print(f"  Total unique scaffolds: {len(scaffold_mapping)}")
    print(f"  Training scaffolds: {len(train_scaffolds)}")
    print(f"  Test scaffolds: {len(test_scaffolds)}")
    print(f"  Scaffold overlap: {len(overlap)} (should be 0)")
    print(f"  Training samples: {len(train_idx)}")
    print(f"  Test samples: {len(test_idx)}")
    print(f"  Test scaffold-group fraction: {len(test_scaffolds) / len(scaffold_mapping):.3f}")
    print(f"  Test compound fraction: {len(test_idx) / len(scaffold_groups):.3f}")
    
    return X_train, X_test, y_train, y_test, scaffold_groups, train_idx, test_idx


def get_scaffold_cv_splits(X, y, smiles, n_splits=5):
    """
    Generate scaffold-based cross-validation splits.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target values
    smiles : list
        SMILES strings
    n_splits : int
        Number of CV folds
        
    Returns:
    --------
    list : List of (train_idx, val_idx) tuples
    numpy.ndarray : Scaffold group labels
    
    Note: GroupKFold is deterministic — splits depend only on group assignments,
    not on any random seed.
    """
    scaffold_groups, _ = get_scaffold_groups(smiles)
    
    # Use GroupKFold for scaffold-aware cross-validation
    gkf = GroupKFold(n_splits=n_splits)
    
    splits = list(gkf.split(X, y, groups=scaffold_groups))
    
    return splits, scaffold_groups


def analyze_scaffold_distribution(smiles_list, y_values, threshold_active=6.0):
    """
    Analyze scaffold distribution and activity patterns.
    
    Parameters:
    -----------
    smiles_list : list
        List of SMILES strings
    y_values : array-like
        pIC50 values
    threshold_active : float
        pIC50 threshold for active compounds
        
    Returns:
    --------
    dict : Scaffold statistics
    """
    scaffold_groups, scaffold_mapping = get_scaffold_groups(smiles_list)
    
    # Count compounds per scaffold
    scaffold_counts = defaultdict(int)
    scaffold_activities = defaultdict(list)
    
    for i, (smiles, y) in enumerate(zip(smiles_list, y_values)):
        scaffold = get_scaffold(smiles)
        scaffold_counts[scaffold] += 1
        scaffold_activities[scaffold].append(y)
    
    # Classify compounds using threshold_active
    total_compounds = sum(scaffold_counts.values())
    active_count = sum(
        1 for activities in scaffold_activities.values()
        for a in activities if a >= threshold_active
    )

    # Calculate statistics
    stats = {
        'total_scaffolds': len(scaffold_mapping),
        'singleton_scaffolds': sum(1 for c in scaffold_counts.values() if c == 1),
        'max_scaffold_size': max(scaffold_counts.values()),
        'mean_scaffold_size': np.mean(list(scaffold_counts.values())),
        'median_scaffold_size': np.median(list(scaffold_counts.values())),
        'active_compounds': active_count,
        'inactive_compounds': total_compounds - active_count,
        'active_ratio': active_count / total_compounds if total_compounds > 0 else 0,
    }
    
    # Top scaffolds by frequency
    top_scaffolds = sorted(scaffold_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    stats['top_scaffolds'] = top_scaffolds
    
    return stats


if __name__ == "__main__":
    # Test 1: Scaffold grouping
    test_smiles = [
        "c1ccccc1",  # benzene
        "c1ccc(O)cc1",  # phenol
        "c1ccc(N)cc1",  # aniline
        "c1ccc2ccccc2c1",  # naphthalene
        "c1ccc2ccccc2c1",  # naphthalene (duplicate scaffold)
    ]
    
    groups, mapping = get_scaffold_groups(test_smiles)
    print("Test 1 — Scaffold grouping:")
    for smi, grp in zip(test_smiles, groups):
        print(f"  {smi} -> Group {grp}")
    print(f"  Unique scaffolds: {len(mapping)}")

    # Test 2: Scaffold split produces zero overlap
    split_smiles = [
        "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccc(F)cc1",
        "c1ccc2ccccc2c1", "c1ccc2cc(O)ccc2c1", "c1ccc2cc(N)ccc2c1",
        "c1ccncc1", "c1ccnc(N)c1", "c1ccnc(O)c1",
        "c1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccc(O)cc2)cc1",
        "c1ccc(-c2ccc(N)cc2)cc1", "c1ccc(-c2ccc(F)cc2)cc1",
        "C1CCCCC1", "C1CCCC(O)C1", "C1CCCC(N)C1", "C1CCCC(F)C1",
        "c1ccc(CC)cc1", "c1ccc(CCO)cc1",
    ]
    X_dummy = np.random.RandomState(0).rand(len(split_smiles), 5)
    y_dummy = np.random.RandomState(0).rand(len(split_smiles))

    X_tr, X_te, y_tr, y_te, grps, tr_idx, te_idx = scaffold_split(
        X_dummy, y_dummy, split_smiles, test_size=0.3, random_state=42,
    )
    train_scaffolds = set(grps[tr_idx])
    test_scaffolds = set(grps[te_idx])
    overlap = train_scaffolds & test_scaffolds
    assert len(overlap) == 0, f"Scaffold overlap detected: {overlap}"
    print("\nTest 2 — Scaffold split: zero overlap confirmed")

    # Test 3: Fallback counter
    reset_scaffold_fallback_count()
    get_scaffold("INVALID_SMILES_XYZ")
    assert get_scaffold_fallback_count() == 1, "Fallback counter did not increment"
    reset_scaffold_fallback_count()
    print("Test 3 — Fallback counter: working correctly")

    print("\nAll self-tests passed.")
