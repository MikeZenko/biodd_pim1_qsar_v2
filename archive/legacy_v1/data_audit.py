"""Create a machine-readable provenance and curation audit for the frozen PIM1 data.

This script deliberately reads the committed CSV exports rather than downloading from
the live ChEMBL API.  It therefore documents the exact data used by the manuscript.
"""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "CHEMBL2147"
RAW_PATH = DATA_DIR / "bioactivity_data_raw.csv"
PREPROCESSED_PATH = DATA_DIR / "bioactivity_data_preprocessed.csv"
MODELING_PATH = DATA_DIR / "pancreatic_cancer_bioactivity_data_3class_pIC50_morgan_fp.csv"
CAP_NM = 100_000_000.0


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def counter_as_dict(counter):
    return {str(key): int(value) for key, value in counter.most_common()}


def main():
    for path in (RAW_PATH, PREPROCESSED_PATH, MODELING_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen data file: {path}")

    raw_rows = read_rows(RAW_PATH)
    preprocessed_rows = read_rows(PREPROCESSED_PATH)
    modeling_rows = read_rows(MODELING_PATH)

    values_by_pair = defaultdict(list)
    pic50_by_pair = defaultdict(list)
    high_records = []
    assay_counts_high = Counter()
    unit_counts = Counter()
    relation_counts = Counter()
    type_counts = Counter()
    flag_counts = Counter()
    target_counts = Counter()
    values = []

    for row in raw_rows:
        target_counts[row.get("target_chembl_id", "")] += 1
        type_counts[row.get("standard_type", "")] += 1
        relation_counts[row.get("standard_relation", "")] += 1
        unit_counts[row.get("standard_units", "")] += 1
        flag_counts[row.get("standard_flag", "")] += 1

        value = float(row["standard_value"])
        values.append(value)
        pair = (row["molecule_chembl_id"], row["target_chembl_id"])
        values_by_pair[pair].append(value)

        pic50 = -math.log10(value * 1e-9)
        pic50_by_pair[pair].append(pic50)
        if pic50 > 10:
            high_records.append(row)
            assay_counts_high[row.get("assay_chembl_id", "unknown")] += 1

    median_pic50_by_pair = {}
    for pair, pair_values in pic50_by_pair.items():
        ordered = sorted(pair_values)
        mid = len(ordered) // 2
        median_value = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        # pair_values already contain record-level pIC50 values.  The modeling
        # workflow aggregates these directly, so do not transform a second time.
        median_pic50_by_pair[pair] = median_value

    high_median_pairs = [pair for pair, value in median_pic50_by_pair.items() if value > 10]
    high_median_single_measurement = sum(
        len(values_by_pair[pair]) == 1 for pair in high_median_pairs
    )

    # Independently reconcile the audit aggregation against the frozen modeling file.
    # This catches accidental changes such as applying the pIC50 transform twice.
    modeling_by_pair = {
        (row["molecule_chembl_id"], row["target_chembl_id"]): float(row["pIC50"])
        for row in modeling_rows
    }
    if len(modeling_by_pair) != len(median_pic50_by_pair):
        raise RuntimeError(
            "Raw aggregation and modeling export have different molecule-target pair counts: "
            f"{len(median_pic50_by_pair)} versus {len(modeling_by_pair)}."
        )
    mismatches = [
        pair for pair, value in median_pic50_by_pair.items()
        if pair not in modeling_by_pair or not math.isclose(value, modeling_by_pair[pair], abs_tol=1e-8)
    ]
    if mismatches:
        raise RuntimeError(
            f"Raw pIC50 aggregation does not reproduce {len(mismatches)} modeling-file rows."
        )

    checksums = {
        path.name: sha256(path)
        for path in (RAW_PATH, PREPROCESSED_PATH, MODELING_PATH)
    }
    audit = {
        "purpose": "Provenance and curation audit for the frozen local PIM1 dataset used by the manuscript.",
        "source_note": (
            "This records the checked-in local export. The download script queries the live ChEMBL REST API "
            "and should not be used to reproduce the v33 freeze without a separately archived v33 export."
        ),
        "files_sha256": checksums,
        "record_counts": {
            "raw_activity_records": len(raw_rows),
            "preprocessed_activity_records": len(preprocessed_rows),
            "modeling_molecule_target_medians": len(modeling_rows),
            "unique_molecule_target_pairs_from_raw": len(values_by_pair),
        },
        "raw_filter_audit": {
            "target_chembl_id": counter_as_dict(target_counts),
            "standard_type": counter_as_dict(type_counts),
            "standard_relation": counter_as_dict(relation_counts),
            "standard_units": counter_as_dict(unit_counts),
            "standard_flag": counter_as_dict(flag_counts),
        },
        "upper_ic50_cap": {
            "cap_nm": CAP_NM,
            "raw_maximum_nm": max(values),
            "records_affected": int(sum(value > CAP_NM for value in values)),
        },
        "high_potency_tail": {
            "definition": "pIC50 > 10 (nominal IC50 < 100 pM)",
            "raw_measurements": len(high_records),
            "molecule_target_medians": len(high_median_pairs),
            "molecule_target_medians_with_one_raw_measurement": int(high_median_single_measurement),
            "reconciled_to_modeling_export": True,
            "top_assays_by_high_potency_measurement": counter_as_dict(assay_counts_high),
        },
        "deduplication": {
            "unit_of_aggregation": "molecule_chembl_id and target_chembl_id",
            "endpoint_aggregation": "median record-level pIC50 per molecule-target pair",
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for output_path in (DATA_DIR / "data_checksums.json", RESULTS_DIR / "data_curation_audit.json"):
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2)
            handle.write("\n")

    summary_path = RESULTS_DIR / "data_curation_summary.csv"
    summary_rows = [
        ("raw_activity_records", len(raw_rows)),
        ("modeling_molecule_target_medians", len(modeling_rows)),
        ("raw_maximum_ic50_nm", max(values)),
        ("upper_cap_records_affected", sum(value > CAP_NM for value in values)),
        ("high_potency_raw_measurements", len(high_records)),
        ("high_potency_molecule_target_medians", len(high_median_pairs)),
        ("high_potency_single_measurement_medians", high_median_single_measurement),
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        writer.writerows(summary_rows)

    print(f"Wrote {DATA_DIR / 'data_checksums.json'}")
    print(f"Wrote {RESULTS_DIR / 'data_curation_audit.json'}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
