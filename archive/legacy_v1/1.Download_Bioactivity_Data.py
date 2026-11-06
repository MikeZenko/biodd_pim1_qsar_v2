#Importing libraries
import pandas as pd
import numpy as np
import os
import sys
import urllib.request
import urllib.parse
import time
import json
import ssl
import hashlib
import warnings
from datetime import datetime, timezone
from tqdm import tqdm
from pathlib import Path

print("Step 1: Downloading Enhanced Pancreatic Cancer Bioactivity Data from ChEMBL")

# Resolve project root and data directory
# Use the script's directory as the project root
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# SSL context - try verified first, fall back to unverified with warning
try:
    ssl_context = ssl.create_default_context()
except Exception:
    warnings.warn("Could not create verified SSL context, falling back to unverified. "
                  "This is a security risk - consider fixing SSL certificates.")
    ssl_context = ssl._create_unverified_context()

# ChEMBL API endpoints
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

def get_chembl_data(endpoint, params=None):
    """Get data from ChEMBL REST API with retry logic using urllib"""
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{CHEMBL_API_BASE}/{endpoint}?{query_string}"
    else:
        url = f"{CHEMBL_API_BASE}/{endpoint}"
        
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            req = urllib.request.Request(url)
            req.add_header('Accept', 'application/json')
            
            with urllib.request.urlopen(req, context=ssl_context) as response:
                if response.getcode() == 200:
                    data = response.read().decode('utf-8')
                    return json.loads(data)
                else:
                    raise Exception(f"HTTP Error: {response.getcode()}")
        except Exception as e:
            retry_count += 1
            print(f"Attempt {retry_count} failed: {e}")
            if retry_count < max_retries:
                sleep_time = 3 * retry_count
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Failed to download data after multiple attempts")
                raise

def get_all_compounds(target_chembl_id, activity_type="IC50"):
    """Get all compounds for a target using pagination"""
    print(f"Retrieving all {activity_type} bioactivity data for target {target_chembl_id}...")
    
    all_activities = []
    limit = 100  # Number of records per page
    offset = 0
    total_count = None
    
    params = {
        "target_chembl_id": target_chembl_id,
        "standard_type": activity_type,
        "standard_relation": "=",
        "standard_units": "nM",
        "standard_flag": "1",
        "standard_value__isnull": "false",
        "limit": limit,
        "offset": offset
    }
    
    # First request to get total count
    try:
        initial_data = get_chembl_data("activity", params)
        total_count = initial_data.get('page_meta', {}).get('total_count', 0)
        if 'activities' in initial_data:
            all_activities.extend(initial_data['activities'])
        
        print(f"Found {total_count} {activity_type} activity data points for {target_chembl_id}")
        
        # Fetch remaining pages
        if total_count > limit:
            pbar = tqdm(total=total_count, initial=len(all_activities), desc=f"Downloading {target_chembl_id} data")
            
            while len(all_activities) < total_count:
                offset += limit
                params["offset"] = offset
                page_data = get_chembl_data("activity", params)
                if 'activities' in page_data:
                    new_activities = page_data['activities']
                    all_activities.extend(new_activities)
                    pbar.update(len(new_activities))
                else:
                    break
            
            pbar.close()
    
    except Exception as e:
        print(f"Error fetching data for {target_chembl_id}: {e}")
    
    return all_activities

def get_target_info(target_chembl_id):
    """Get information about a target"""
    try:
        target_info = get_chembl_data(f"target/{target_chembl_id}")
        print(f"Target info retrieved: {target_info['target_type']} - {target_info['pref_name']}")
        return target_info
    except Exception as e:
        print(f"Error getting target info for {target_chembl_id}: {e}")
        return None

def validate_target(target_chembl_id, expected_name, target_info):
    """
    Validate that the downloaded target matches expectations.
    This prevents accidental use of wrong ChEMBL IDs.
    """
    if target_info is None:
        print(f"WARNING: Could not validate target {target_chembl_id}")
        return False

    pref_name = target_info.get('pref_name', '').lower()
    expected_lower = expected_name.lower()

    # Check if the expected name appears in the preferred name
    # PIM1 should match "pim-1" or "pim1"
    expected_variants = [
        expected_lower,
        expected_lower.replace('pim', 'pim-'),
    ]

    name_matches = any(variant in pref_name for variant in expected_variants)

    if not name_matches:
        print(f"\n" + "="*60)
        print(f"CRITICAL WARNING: Target mismatch detected!")
        print(f"  ChEMBL ID: {target_chembl_id}")
        print(f"  Expected: {expected_name}")
        print(f"  Actual: {target_info.get('pref_name', 'Unknown')}")
        print(f"  Please verify the ChEMBL ID is correct!")
        print(f"="*60 + "\n")
        return False

    print(f"  Target validated: {target_chembl_id} = {target_info.get('pref_name')}")
    return True

def fetch_smiles_for_molecules(molecule_ids):
    """Fetch SMILES for a list of molecule ChEMBL IDs"""
    print(f"Retrieving SMILES for {len(molecule_ids)} compounds...")
    smiles_dict = {}
    
    for mol_id in tqdm(molecule_ids, desc="Fetching SMILES"):
        try:
            mol_data = get_chembl_data(f"molecule/{mol_id}")
            if 'molecule_structures' in mol_data and mol_data['molecule_structures'] is not None:
                if 'canonical_smiles' in mol_data['molecule_structures']:
                    smiles_dict[mol_id] = mol_data['molecule_structures']['canonical_smiles']
        except Exception as e:
            print(f"Error getting SMILES for {mol_id}: {e}")
            continue
            
    return smiles_dict

def process_activities(activities, target_name):
    """Process activity data into a DataFrame with target information"""
    if not activities:
        return pd.DataFrame()
    
    df = pd.DataFrame(activities)
    
    # Add target name column
    df['target_name'] = target_name
    df['target_chembl_id'] = activities[0].get('target_chembl_id', '')
    
    # Check for required columns
    required_columns = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: Missing columns in data: {missing_columns}")
        
        # If SMILES is missing, try to get it
        if 'canonical_smiles' in missing_columns and 'molecule_chembl_id' in df.columns:
            # Get unique molecule IDs - fetch all to ensure data integrity
            molecule_ids = df['molecule_chembl_id'].unique()
            print(f"Fetching SMILES for {len(molecule_ids)} unique molecules...")
            smiles_dict = fetch_smiles_for_molecules(molecule_ids)
            
            df['canonical_smiles'] = df['molecule_chembl_id'].map(smiles_dict)
    
    return df

def preprocess_bioactivity_data(df):
    """Apply preprocessing steps to the bioactivity data"""
    if df.empty:
        return df
    
    print("Pre-processing bioactivity data...")
    
    # Keep only entries with the necessary data
    df2 = df.dropna(subset=['standard_value', 'canonical_smiles', 'molecule_chembl_id']).copy()
    print(f"After removing entries with missing data: {len(df2)}")

    # These two checks mirror the API filters and protect against unexpected
    # API behaviour or incomplete exports.  The model assumes nanomolar IC50
    # values and uses only ChEMBL-standardized measurements.
    if 'standard_units' in df2.columns:
        before = len(df2)
        df2 = df2[df2['standard_units'].astype(str).str.lower() == 'nm'].copy()
        print(f"After retaining standard_units = nM: {len(df2)} (removed {before - len(df2)})")
    else:
        print("WARNING: standard_units is absent; cannot verify nM units.")

    if 'standard_flag' in df2.columns:
        before = len(df2)
        valid_flags = {'1', 'true', 'True'}
        df2 = df2[df2['standard_flag'].astype(str).isin(valid_flags)].copy()
        print(f"After retaining standard_flag = 1: {len(df2)} (removed {before - len(df2)})")
    else:
        print("WARNING: standard_flag is absent; cannot verify standardized records.")
    
    # Convert standard_value to numeric, handling any non-numeric values
    df2['standard_value'] = pd.to_numeric(df2['standard_value'], errors='coerce')
    df2 = df2.dropna(subset=['standard_value'])
    print(f"After converting to numeric values: {len(df2)}")
    
    # Classify compounds by activity
    bioactivity_class = []
    for i in df2['standard_value']:
        if float(i) >= 10000:
            bioactivity_class.append("inactive")
        elif float(i) <= 1000:
            bioactivity_class.append("active")
        else:
            bioactivity_class.append("intermediate")
    
    # Combine data — reset index to ensure alignment after dropna
    selection = ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'target_name', 'target_chembl_id']
    df3 = df2[selection].reset_index(drop=True)
    df3['bioactivity_class'] = bioactivity_class
    
    # Calculate pIC50
    def norm_value(input_df):
        cap_value_nm = 100000000
        capped = input_df['standard_value'] > cap_value_nm
        print(f"Values capped at {cap_value_nm:.0e} nM: {int(capped.sum())}")
        input_df['standard_value_norm'] = input_df['standard_value'].clip(upper=cap_value_nm)
        return input_df
    
    def pIC50(input_df):
        pIC50_values = []
        for i in input_df['standard_value_norm']:
            molar = i*(10**-9)  # Converts nM to M
            pIC50_values.append(-np.log10(molar))
        input_df['pIC50'] = pIC50_values
        return input_df
    
    df4 = df3.copy()
    df5 = norm_value(df4)
    df6 = pIC50(df5)
    
    return df6

def compute_file_checksum(filepath):
    """Compute SHA-256 checksum of a file for data provenance tracking."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_target_information(targets_data):
    """Save target information to a CSV file"""
    targets_df = pd.DataFrame(targets_data)
    target_path = DATA_DIR / 'target_information.csv'
    targets_df.to_csv(target_path, index=False)
    print(f"Saved target information to {target_path}")

try:
    # Define target ChEMBL IDs
    # CRITICAL: These IDs must match the actual targets:
    #   - CHEMBL2147: Serine/threonine-protein kinase Pim-1 (PIM1)
    # Verified against ChEMBL database - do NOT change without verification
    targets = [
        {"id": "CHEMBL2147", "name": "PIM1"},  # Proto-oncogene serine/threonine-protein kinase Pim-1
    ]
    
    # Store target information
    target_info_list = []
    
    # Get bioactivity data for all targets
    all_activities_df = pd.DataFrame()
    
    for target in targets:
        target_id = target["id"]
        target_name = target["name"]

        # Get target information
        target_info = get_target_info(target_id)

        # Validate that the target matches expectations
        if target_info:
            is_valid = validate_target(target_id, target_name, target_info)
            if not is_valid:
                print(f"Skipping target {target_id} due to validation failure.")
                print("Please verify the ChEMBL ID and update the code if necessary.")
                continue

            target_info_list.append({
                "target_chembl_id": target_id,
                "target_name": target_name,
                "target_type": target_info.get('target_type', ''),
                "organism": target_info.get('organism', ''),
                "pref_name": target_info.get('pref_name', '')
            })
        
        # Get activity data for this target
        activities = get_all_compounds(target_id)
        
        if activities:
            # Process activities into DataFrame
            target_df = process_activities(activities, target_name)
            
            # Append to the all activities dataframe
            all_activities_df = pd.concat([all_activities_df, target_df], ignore_index=True)
            
            print(f"Retrieved {len(target_df)} compounds for {target_name}")
        else:
            print(f"No activity data found for {target_name}")
    
    # Save raw combined data
    if not all_activities_df.empty:
        raw_path = DATA_DIR / 'bioactivity_data_raw.csv'
        all_activities_df.to_csv(raw_path, index=False)
        print(f"Saved raw data to {raw_path} with {len(all_activities_df)} total compounds")
        
        # Save target information
        save_target_information(target_info_list)
        
        # Preprocess the data
        processed_df = preprocess_bioactivity_data(all_activities_df)
        
        # Save processed data
        processed_path = DATA_DIR / 'bioactivity_data_preprocessed.csv'
        processed_df.to_csv(processed_path, index=False)
        print(f"Saved preprocessed data to {processed_path} with {len(processed_df)} compounds")
        
        # Compute and save provenance metadata + checksums
        file_checksums = {}
        for file in ['bioactivity_data_raw.csv', 'bioactivity_data_preprocessed.csv', 'target_information.csv']:
            file_path = DATA_DIR / file
            if file_path.exists():
                file_checksums[file] = compute_file_checksum(file_path)

        provenance = {
            "query": {
                "api_base": CHEMBL_API_BASE,
                "targets": [
                    {
                        "target_chembl_id": t["id"],
                        "target_name": t["name"],
                        "activity_type": "IC50",
                        "filters": {
                            "standard_relation": "=",
                            "standard_value__isnull": "false",
                        },
                    }
                    for t in targets
                ],
                "query_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "file_checksums_sha256": file_checksums,
            "total_raw_compounds": len(all_activities_df),
            "total_preprocessed_compounds": len(processed_df),
        }

        provenance_path = DATA_DIR / 'data_checksums.json'
        with open(provenance_path, 'w') as f:
            json.dump(provenance, f, indent=2)
        print(f"Saved data provenance to {provenance_path}")

        # Show file information
        print("\nFiles generated:")
        for file in ['bioactivity_data_raw.csv', 'bioactivity_data_preprocessed.csv', 'target_information.csv']:
            file_path = DATA_DIR / file
            if file_path.exists():
                print(f"- {file_path} ({file_path.stat().st_size/1024:.1f} KB)"
                      f"  SHA-256: {file_checksums.get(file, 'N/A')}")
    else:
        print("No activity data could be retrieved for any target")
        sys.exit(1)
    
except Exception as e:
    print(f"ERROR: {str(e)}")
    print("Failed to download data from ChEMBL.")
    sys.exit(1)

print("\nStep 1 completed. Run 2.Exploratory_Data_Analysis.py next.")
