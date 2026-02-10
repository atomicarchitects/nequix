import tempfile

import pytest
import torch
import torch_geometric
from e3nn import o3

from nequix.torch.layer_norm import RMSLayerNorm
from nequix.torch.model import (
    NequixTorch,
    get_optimizer_param_groups,
    load_model,
    save_model,
    scatter,
)


def dummy_graph():
    return torch_geometric.data.Data(
        n_node=torch.tensor([3]),
        n_edge=torch.tensor([3]),
        n_graph=torch.zeros(3, dtype=torch.int64),
        x=torch.tensor([0, 1, 0], dtype=torch.int64),
        energy=torch.ones((1,)),
        forces=torch.ones((3, 3), dtype=torch.float32),
        positions=torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ]
        ).to(torch.float32),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        edge_attr=torch.zeros((3, 3), dtype=torch.float32),
        cell=torch.eye(3)[None, :, :],
    )


def test_model():
    # small model for testing
    model = NequixTorch(
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
    )

    batch = dummy_graph()
    energy_per_atom, forces, stress = model(
        batch.x,
        batch.positions,
        batch.edge_attr,
        batch.edge_index,
        batch.cell,
        batch.n_node,
        batch.n_edge,
        batch.n_graph,
    )
    energy = scatter(energy_per_atom, batch.n_graph, dim=0, dim_size=batch.n_node.size(0))
    assert energy.shape == batch.energy.shape
    assert forces.shape == batch.forces.shape
    assert stress.shape == batch.cell.shape

    batch = torch_geometric.data.Batch.from_data_list([batch, batch])
    energy_per_atom, forces, stress = model(
        batch.x,
        batch.positions,
        batch.edge_attr,
        batch.edge_index,
        batch.cell,
        batch.n_node,
        batch.n_edge,
        batch.n_graph,
    )
    energy = scatter(energy_per_atom, batch.n_graph, dim=0, dim_size=batch.n_node.size(0))
    assert energy.shape == batch.energy.shape
    assert forces.shape == batch.forces.shape
    assert stress.shape == batch.cell.shape

    batch = dummy_graph()
    batch.cell = None
    energy_per_atom, forces, stress = model(
        batch.x,
        batch.positions,
        batch.edge_attr,
        batch.edge_index,
        None,
        batch.n_node,
        batch.n_edge,
        batch.n_graph,
    )
    energy = scatter(energy_per_atom, batch.n_graph, dim=0, dim_size=batch.n_node.size(0))
    assert energy.shape == batch.energy.shape
    assert forces.shape == batch.forces.shape
    assert stress is None


@pytest.mark.parametrize("centering", [True, False])
def test_layer_norm(centering):
    irreps = o3.Irreps("8x0e + 4x1o + 2x2e")
    layer_norm = RMSLayerNorm(irreps=irreps, centering=centering, std_balance_degrees=True)
    node_input = torch.randn(5, irreps.dim)
    output = layer_norm(node_input)
    assert layer_norm.irreps == irreps
    assert output.shape == node_input.shape


def test_model_save_load():
    # Create config for model initialization
    config = {
        "cutoff": 6.0,
        "atomic_numbers": [1, 6],  # H and C
        "hidden_irreps": "8x0e+8x1o",
        "lmax": 1,
        "n_layers": 2,
        "radial_basis_size": 4,
        "radial_mlp_size": 8,
        "radial_mlp_layers": 2,
        "radial_polynomial_p": 2.0,
        "mlp_init_scale": 4.0,
        "index_weights": False,
        "layer_norm": False,
        "shift": 1.0,
        "scale": 2.0,
        "avg_n_neighbors": 2.0,
        "atom_energies": {1: 1.0, 6: 2.0},
        "kernel": False,
    }

    # Create model using config parameters
    atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    model = NequixTorch(
        n_species=len(config["atomic_numbers"]),
        cutoff=config["cutoff"],
        lmax=config["lmax"],
        hidden_irreps=config["hidden_irreps"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        index_weights=config["index_weights"],
        layer_norm=config["layer_norm"],
        shift=config["shift"],
        scale=config["scale"],
        avg_n_neighbors=config["avg_n_neighbors"],
        atom_energies=atom_energies,
        kernel=config["kernel"],
    )

    batch = dummy_graph()
    original_energy, original_forces, original_stress = model(
        batch.x,
        batch.positions,
        batch.edge_attr,
        batch.edge_index,
        batch.cell,
        batch.n_node,
        batch.n_edge,
        batch.n_graph,
    )

    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
        save_model(tmp_file.name, model, config)
        loaded_model, _ = load_model(tmp_file.name)
        assert model.lmax == loaded_model.lmax
        assert model.cutoff == loaded_model.cutoff
        assert model.n_species == loaded_model.n_species
        assert model.radial_basis_size == loaded_model.radial_basis_size
        assert model.radial_polynomial_p == loaded_model.radial_polynomial_p
        assert model.shift == loaded_model.shift
        assert model.scale == loaded_model.scale
        assert model.layers[0].avg_n_neighbors == loaded_model.layers[0].avg_n_neighbors
        torch.testing.assert_close(model.atom_energies, loaded_model.atom_energies)
        loaded_energy, loaded_forces, loaded_stress = loaded_model(
            batch.x,
            batch.positions,
            batch.edge_attr,
            batch.edge_index,
            batch.cell,
            batch.n_node,
            batch.n_edge,
            batch.n_graph,
        )
        torch.testing.assert_close(original_energy, loaded_energy)
        torch.testing.assert_close(original_forces, loaded_forces)
        torch.testing.assert_close(original_stress, loaded_stress)


def test_weight_decay_mask():
    model = NequixTorch(
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        layer_norm=True,
    )
    params_groups = get_optimizer_param_groups(model, 0.001)

    # First param group should have the weight decay
    assert params_groups[0]["weight_decay"] == 0.001
    # should be weight decay on e3nn linear layers and normal linear weights
    torch.testing.assert_close(params_groups[0]["params"][0], model.layers[0].linear_1.weight)
    torch.testing.assert_close(
        params_groups[0]["params"][1], model.layers[0].radial_mlp.layers[0].weights
    )

    # TODO: Have tests for the params that have a weight_decay
