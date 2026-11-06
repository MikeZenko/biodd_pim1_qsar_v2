from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pim1_qsar.validation.splits import make_temporal_split


def _records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Identity A is known before the cutoff. Its 2020 value must not
            # update the development label.
            {"identity": "A", "year": 2015, "pIC50": 5.0, "scaffold": "s1", "record": "a1"},
            {"identity": "A", "year": 2016, "pIC50": 7.0, "scaffold": "s1", "record": "a2"},
            {"identity": "A", "year": 2020, "pIC50": 100.0, "scaffold": "s1", "record": "a3"},
            {"identity": "B", "year": 2016, "pIC50": 8.0, "scaffold": "s2", "record": "b1"},
            # Identities C and D are first observed after the cutoff.
            {"identity": "C", "year": 2017, "pIC50": 9.0, "scaffold": "s3", "record": "c1"},
            {"identity": "C", "year": 2018, "pIC50": 7.0, "scaffold": "s3", "record": "c2"},
            {"identity": "D", "year": 2020, "pIC50": 6.5, "scaffold": "s4", "record": "d1"},
        ]
    )


def _split(records: pd.DataFrame):
    return make_temporal_split(
        records,
        cutoff_year=2016,
        identity_col="identity",
        year_col="year",
        value_col="pIC50",
        scaffold_col="scaffold",
    )


def test_temporal_split_is_identity_disjoint_and_uses_first_seen_year():
    split = _split(_records())

    assert set(split.development["identity"]) == {"A", "B"}
    assert set(split.temporal_test["identity"]) == {"C", "D"}
    assert set(split.development["identity"]).isdisjoint(split.temporal_test["identity"])
    assert split.development["last_record_year"].max() <= 2016
    assert split.temporal_test["first_record_year"].min() >= 2017


def test_development_label_excludes_post_cutoff_measurements():
    split = _split(_records())
    development = split.development.set_index("identity")

    assert development.loc["A", "pIC50"] == pytest.approx(6.0)
    assert development.loc["A", "measurement_count"] == 2
    assert list(split.cross_cutoff_records["record"]) == ["a3"]


def test_future_value_mutation_cannot_change_development_artifact():
    baseline = _split(_records())
    mutated_records = _records()
    mutated_records.loc[mutated_records["record"] == "a3", "pIC50"] = -999.0
    mutated_records.loc[mutated_records["record"] == "c1", "pIC50"] = 999.0
    mutated = _split(mutated_records)

    pd.testing.assert_frame_equal(baseline.development, mutated.development)
    assert baseline.manifest["development_identity_sha256"] == mutated.manifest[
        "development_identity_sha256"
    ]


def test_split_manifest_is_deterministic_under_record_reordering():
    baseline = _split(_records())
    shuffled = _records().sample(frac=1.0, random_state=73).reset_index(drop=True)
    replay = _split(shuffled)

    pd.testing.assert_frame_equal(baseline.development, replay.development)
    pd.testing.assert_frame_equal(baseline.temporal_test, replay.temporal_test)
    assert baseline.manifest == replay.manifest


@pytest.mark.parametrize(
    "column,bad_value,match",
    [
        ("identity", None, "missing identities"),
        ("year", "unknown", "finite years"),
        ("pIC50", float("inf"), "finite numeric values"),
    ],
)
def test_invalid_record_fields_are_rejected(column, bad_value, match):
    records = _records()
    # pandas 3 rejects incompatible scalar assignment before our validator sees
    # it, so model malformed CSV input with an object-typed column.
    records[column] = records[column].astype(object)
    records.loc[0, column] = bad_value
    with pytest.raises(ValueError, match=match):
        _split(records)


def test_identity_cannot_map_to_multiple_scaffolds():
    records = _records()
    records.loc[records["record"] == "a2", "scaffold"] = "different"
    with pytest.raises(ValueError, match="multiple scaffold"):
        _split(records)
