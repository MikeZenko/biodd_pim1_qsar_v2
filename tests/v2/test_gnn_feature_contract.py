"""Contract tests for the optional, isolated GNN foundation."""

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

pytest.importorskip("rdkit")

from pim1_qsar.models import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    ChempropAdapter,
    ChempropConfig,
    GINEConfig,
    GINERegressor,
    feature_schema,
    featurize_smiles,
    gnn_dependencies_available,
    installed_chemprop_version,
    require_gnn_dependencies,
    set_global_determinism,
    to_pyg_data,
)


def test_canonical_equivalents_have_identical_graph_tensors():
    first = featurize_smiles("CCO")
    second = featurize_smiles("OCC")

    assert first.canonical_smiles == second.canonical_smiles
    np.testing.assert_array_equal(first.atom_features, second.atom_features)
    np.testing.assert_array_equal(first.edge_index, second.edge_index)
    np.testing.assert_array_equal(first.bond_features, second.bond_features)


def test_feature_schema_and_bidirectional_edges_are_stable():
    graph = featurize_smiles("CCO")
    schema = feature_schema()

    assert graph.atom_features.shape == (3, ATOM_FEATURE_DIM)
    assert graph.edge_index.shape == (2, 4)
    assert graph.bond_features.shape == (4, BOND_FEATURE_DIM)
    assert schema["schema_version"] == "gnn_features_v1"
    assert schema["atom_feature_dim"] == ATOM_FEATURE_DIM
    assert schema["bond_feature_dim"] == BOND_FEATURE_DIM

    directed = {
        (int(source), int(target))
        for source, target in graph.edge_index.T
    }
    for source, target in directed:
        assert (target, source) in directed
        forward_index = next(
            index
            for index, edge in enumerate(graph.edge_index.T)
            if tuple(edge) == (source, target)
        )
        reverse_index = next(
            index
            for index, edge in enumerate(graph.edge_index.T)
            if tuple(edge) == (target, source)
        )
        np.testing.assert_array_equal(
            graph.bond_features[forward_index],
            graph.bond_features[reverse_index],
        )


def test_single_atom_and_invalid_smiles_contracts():
    chloride = featurize_smiles("[Cl-]")
    assert chloride.atom_features.shape == (1, ATOM_FEATURE_DIM)
    assert chloride.edge_index.shape == (2, 0)
    assert chloride.bond_features.shape == (0, BOND_FEATURE_DIM)

    with pytest.raises(ValueError, match="could not parse"):
        featurize_smiles("not-a-smiles")


def test_chemprop_adapter_validates_alignment_without_importing_chemprop():
    adapter = ChempropAdapter(ChempropConfig(seed=37))
    batch = adapter.prepare_inputs(
        row_ids=["row-a", "row-b"],
        smiles=["OCC", "c1ccccc1"],
        targets=[6.2, 7.1],
        descriptors=[[1.0, 2.0], [3.0, 4.0]],
    )
    contract = adapter.training_contract(batch)

    assert batch.canonical_smiles[0] == "CCO"
    assert batch.targets.shape == (2, 1)
    assert batch.descriptor_dim == 2
    assert not batch.targets.flags.writeable
    assert contract["n_rows"] == 2
    assert contract["descriptor_dim"] == 2
    assert contract["target_scaling"] == "fit_on_training_partition_only"
    assert "--message-hidden-dim" in adapter.config.cli_arguments()

    with pytest.raises(ValueError, match="unique"):
        adapter.prepare_inputs(
            row_ids=["duplicate", "duplicate"],
            smiles=["CC", "CCC"],
            targets=[1.0, 2.0],
        )


def test_optional_dependency_error_is_actionable_when_gnn_runtime_is_absent():
    if gnn_dependencies_available():
        pytest.skip("The isolated GNN dependencies are installed")
    with pytest.raises(ImportError, match="isolated .venv-gnn"):
        require_gnn_dependencies()
    with pytest.raises(ImportError, match="isolated .venv-gnn"):
        GINERegressor(GINEConfig())


def test_gine_forward_contract_when_optional_dependencies_are_present():
    if not gnn_dependencies_available():
        pytest.skip("Torch/PyG are intentionally absent from the core runtime")

    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Batch

    set_global_determinism(13)
    graphs = [
        to_pyg_data(
            featurize_smiles("CCO"),
            target=6.0,
            descriptors=[1.0, -1.0],
            row_id="a",
        ),
        to_pyg_data(
            featurize_smiles("c1ccccc1"),
            target=7.0,
            descriptors=[0.5, 0.25],
            row_id="b",
        ),
    ]
    batch = Batch.from_data_list(graphs)
    model = GINERegressor(
        GINEConfig(
            hidden_dim=32,
            num_layers=2,
            ffn_hidden_dim=16,
            descriptor_dim=2,
        )
    )
    model.eval()
    with torch.no_grad():
        prediction = model(batch)

    assert tuple(prediction.shape) == (2,)
    assert torch.isfinite(prediction).all()


def test_installed_chemprop_matches_the_audited_version_if_present():
    installed = installed_chemprop_version()
    if installed is None:
        pytest.skip("Chemprop is intentionally absent from the core runtime")
    assert installed == "2.2.4"
