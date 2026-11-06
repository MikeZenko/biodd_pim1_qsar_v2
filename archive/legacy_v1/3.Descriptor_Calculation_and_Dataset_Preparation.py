#Load bioactivity data and calculate Morgan fingerprints (ECFP6) and physicochemical descriptors
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolSurf, GraphDescriptors

print("Step 3: Descriptor Calculation and Dataset Preparation")

# Resolve project root and data directory
# Use the script's directory as the project root
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# Load the preprocessed dataset from Step 2
input_file = DATA_DIR / 'pancreatic_cancer_bioactivity_data_2class_lipinski.csv'
if not os.path.exists(input_file):
    print(f"ERROR: Input file {input_file} not found! Run Step 2 first.")
    sys.exit(1)
    
df3 = pd.read_csv(input_file)
print(f"Loaded data with {len(df3)} compounds and {df3.shape[1]} features")

# Deduplicate molecules: same molecule may have multiple pIC50 measurements
# Take median pIC50 for duplicates to reduce label noise
print("\nDeduplicating molecules with multiple measurements...")
original_count = len(df3)
duplicates_count = df3.groupby(['molecule_chembl_id', 'target_chembl_id']).size()
duplicates_count = duplicates_count[duplicates_count > 1]
print(f"  Molecules with multiple measurements: {len(duplicates_count)}")

# Group by molecule and target, aggregate pIC50 by median
agg_dict = {col: 'first' for col in df3.columns if col not in ['pIC50', 'molecule_chembl_id', 'target_chembl_id']}
agg_dict['pIC50'] = 'median'
df3 = df3.groupby(['molecule_chembl_id', 'target_chembl_id'], as_index=False).agg(agg_dict)
print(f"  After deduplication: {len(df3)} unique molecule-target pairs (removed {original_count - len(df3)} duplicates)")

print(f"Total molecules for descriptor calculation: {len(df3)}")

# Calculate multiple descriptor types using RDKit
print("Calculating molecular descriptors...")

# Lists to store fingerprint and descriptor data
fingerprint_data = []
successful_indices = []
failed_mols = 0

# Morgan fingerprint parameters
# UPDATED: Changed from ECFP4 (radius=2, 1024 bits) to ECFP6 (radius=3, 2048 bits)
# to match paper specifications and capture more structural detail
radius = 3  # ECFP6 (Extended Connectivity Fingerprint with radius 3)
nBits = 2048  # Number of bits (industry standard for ECFP6)

# Dictionary to map physicochemical descriptor names to their calculation functions
physchem_descriptors = {
    # Basic properties
    'MolWt': Descriptors.MolWt,
    'LogP': Descriptors.MolLogP,
    'NumHDonors': Lipinski.NumHDonors,
    'NumHAcceptors': Lipinski.NumHAcceptors,
    'NumRotatableBonds': Lipinski.NumRotatableBonds,
    'NumAromaticRings': Lipinski.NumAromaticRings,
    'NumHeteroatoms': Lipinski.NumHeteroatoms,
    'FractionCSP3': Lipinski.FractionCSP3,
    
    # Surface properties
    'TPSA': MolSurf.TPSA,
    'LabuteASA': MolSurf.LabuteASA,
    
    # Topological properties
    'BalabanJ': GraphDescriptors.BalabanJ,
    'BertzCT': GraphDescriptors.BertzCT,
    'Chi0v': GraphDescriptors.Chi0v,
    'Chi1v': GraphDescriptors.Chi1v,
    'Chi2v': GraphDescriptors.Chi2v,
    'Chi3v': GraphDescriptors.Chi3v,
    'Chi4v': GraphDescriptors.Chi4v,
    
    # Drug-likeness properties
    'QED': Descriptors.qed,
    'NumRings': Descriptors.RingCount,
    'HeavyAtomCount': Descriptors.HeavyAtomCount,
    'FormalCharge': lambda mol: sum([atom.GetFormalCharge() for atom in mol.GetAtoms()]),
    'NHOHCount': Descriptors.NHOHCount,
    'NOCount': Descriptors.NOCount
}

for idx, row in df3.iterrows():
    try:
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        
        if mol is not None:
            # Calculate Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            
            # Convert fingerprint to a list of 0s and 1s
            fp_bits = list(fp.ToBitString())
            fp_bits = [int(b) for b in fp_bits]
            
            # Create a dictionary with fingerprint bits and physicochemical descriptors
            record = {}
            
            # Add Morgan fingerprint bits
            for i, bit in enumerate(fp_bits):
                record[f'Morgan_{i}'] = bit
                
            # Add physicochemical descriptors
            for desc_name, desc_func in physchem_descriptors.items():
                try:
                    value = desc_func(mol)
                    record[desc_name] = value
                except Exception as e:
                    # Log the specific error for debugging
                    record[desc_name] = np.nan
            
            fingerprint_data.append(record)
            successful_indices.append(idx)
        else:
            print(f"Warning: Could not parse SMILES: {row['canonical_smiles']}")
            failed_mols += 1
    except Exception as e:
        print(f"Error processing molecule {row['molecule_chembl_id']}: {e}")
        failed_mols += 1

print(f"Successfully calculated descriptors for {len(fingerprint_data)} compounds")
print(f"Failed to process {failed_mols} compounds")

# Create descriptor DataFrame aligned to successfully parsed molecules
df3_X = pd.DataFrame(fingerprint_data, index=successful_indices)
df3 = df3.loc[successful_indices]

# Handle missing values
# NOTE: We do NOT impute here to avoid data leakage. Imputation will be done
# in the modeling scripts (Step 5) with proper train/test separation -
# imputer is fit on training data only and applied to test data.
missing_count = df3_X.isna().sum().sum()
print(f"Missing values in descriptors: {missing_count}")
if missing_count > 0:
    print("  (These will be imputed in Step 5 using train-only statistics)")

# Print summary of descriptor types
morgan_count = sum(1 for col in df3_X.columns if col.startswith('Morgan_'))
physchem_count = sum(1 for col in df3_X.columns if not col.startswith('Morgan_'))
print(f"Generated {morgan_count} Morgan fingerprint bits and {physchem_count} physicochemical descriptors")
print(f"Total descriptors: {df3_X.shape[1]}")
print(f"Descriptor matrix shape: {df3_X.shape}")

# Get the pIC50 values as the target variable
df3_Y = df3['pIC50']
print(f"Target variable: pIC50 with {len(df3_Y)} values")

# Combine the X and Y data into a single DataFrame
context_columns = [
    col for col in ['target_name', 'target_chembl_id', 'canonical_smiles', 'molecule_chembl_id']
    if col in df3.columns
]
dataset3 = pd.concat([df3[context_columns], df3_X, df3_Y], axis=1)
print(f"Final dataset has {dataset3.shape[0]} compounds and {dataset3.shape[1]} features (including pIC50)")

# Save the final dataset for model building
# NOTE: Using morgan_fp in filename to correctly reflect that we use Morgan/ECFP6 fingerprints
output_file = DATA_DIR / 'pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv'
dataset3.to_csv(output_file, index=False)
print(f"Saved final dataset to {output_file}")
print(f"Fingerprint type: Morgan (ECFP6, radius=3, 2048 bits)")

# List files created for the next step
created_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(DATA_DIR / f) and f.endswith('.csv')]
print("\nFiles generated for model building:")
for file in created_files:
    file_path = DATA_DIR / file
    if file_path == output_file:
        print(f"- {file_path} ({file_path.stat().st_size/1024/1024:.2f} MB) <- Use this for modeling")
    elif file.endswith('.csv'):
        print(f"- {file_path} ({file_path.stat().st_size/1024/1024:.2f} MB)")

print("\nStep 3 completed. Run 4.Regression_Models_with_Random_Forest.py next.")


