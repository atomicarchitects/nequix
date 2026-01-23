import tempfile

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import pytest

from nequix.layer_norm import RMSLayerNorm
from nequix.model import Nequix, OEQ_AVAILABLE, load_model, save_model, weight_decay_mask


def dummy_graph():
    return jraph.GraphsTuple(
        n_node=np.array([3]),
        n_edge=np.array([3]),
        nodes={
            "species": np.array([0, 1, 0], dtype=np.int32),
            "positions": np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ]
            ).astype(np.float32),
            "forces": np.ones((3, 3), dtype=np.float32),
        },
        edges={
            "shifts": np.zeros((3, 3), dtype=np.float32),
        },
        senders=jnp.array([0, 1, 2]),
        receivers=jnp.array([1, 2, 0]),
        globals={"energy": np.ones((1,)), "cell": np.eye(3)[None, ...]},
    )


def test_model():
    key = jax.random.key(0)

    # small model for testing
    model = Nequix(
        key,
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
    )

    batch = dummy_graph()
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=4, n_graph=2)
    energy, forces, stress = model(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress.shape == batch_padded.globals["cell"].shape

    energy, forces, stress = eqx.filter_jit(model)(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress.shape == batch_padded.globals["cell"].shape

    batch = jraph.batch_np([batch, batch])
    batch_padded = jraph.pad_with_graphs(batch, n_node=7, n_edge=7, n_graph=3)
    energy, forces, stress = model(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress.shape == batch_padded.globals["cell"].shape

    # check that stress is None if cell is None
    batch = dummy_graph()
    batch = batch._replace(globals={**batch.globals, "cell": None})
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=4, n_graph=2)
    energy, forces, stress = model(batch_padded)
    assert energy.shape == batch_padded.globals["energy"].shape
    assert forces.shape == batch_padded.nodes["forces"].shape
    assert stress is None


@pytest.mark.parametrize("centering", [True, False])
def test_layer_norm(centering):
    irreps = e3nn.Irreps("8x0e + 4x1o + 2x2e")
    layer_norm = RMSLayerNorm(irreps=irreps, centering=centering, std_balance_degrees=True)
    x = jax.random.normal(jax.random.key(0), (5, irreps.dim))
    node_input = e3nn.IrrepsArray(irreps, x)
    output = layer_norm(node_input)
    assert output.irreps == node_input.irreps
    assert output.shape == node_input.shape


def test_model_save_load():
    key = jax.random.key(0)

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
        "index_weights": True,
        "layer_norm": False,
        "shift": 1.0,
        "scale": 2.0,
        "avg_n_neighbors": 2.0,
        "atom_energies": {1: 1.0, 6: 2.0},
    }

    # Create model using config parameters
    atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    model = Nequix(
        key,
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
    )

    batch = dummy_graph()
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=3, n_graph=2)

    original_energy, original_forces, original_stress = model(batch_padded)

    with tempfile.NamedTemporaryFile(suffix=".eqx") as tmp_file:
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
        np.testing.assert_allclose(model.atom_energies, loaded_model.atom_energies)
        loaded_energy, loaded_forces, loaded_stress = loaded_model(batch_padded)
        np.testing.assert_allclose(original_energy, loaded_energy)
        np.testing.assert_allclose(original_forces, loaded_forces)
        np.testing.assert_allclose(original_stress, loaded_stress)


def test_weight_decay_mask():
    key = jax.random.key(0)
    model = Nequix(
        key,
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        layer_norm=True,
    )
    mask = weight_decay_mask(model)
    # should be weight decay on e3nn linear layers and normal linear weights
    assert all(mask.layers[0].linear_1._weights.values())
    assert mask.layers[0].radial_mlp.layers[0].weights

    # no weight decay on atom energies, biases, or layer norms
    assert not mask.atom_energies
    assert not mask.layers[0].radial_mlp.layers[0].bias
    assert not any(mask.layers[0].layer_norm.affine_weight)
    assert not mask.layers[0].layer_norm.affine_bias


@pytest.mark.skipif(not OEQ_AVAILABLE, reason="OpenEquivariance not installed")
def test_model_with_openequivariance_kernel():
    """Test model with OpenEquivariance kernels."""
    key = jax.random.key(0)

    model_no_kernel = Nequix(
        key,
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        kernel=False,
    )

    model_kernel = Nequix(
        key,
        n_species=2,
        lmax=1,
        hidden_irreps="8x0e+8x1o",
        n_layers=2,
        radial_basis_size=4,
        radial_mlp_size=8,
        radial_mlp_layers=2,
        kernel=True,
    )

    batch = dummy_graph()
    batch_padded = jraph.pad_with_graphs(batch, n_node=4, n_edge=4, n_graph=2)

    energy_no_kernel, forces_no_kernel, stress_no_kernel = model_no_kernel(batch_padded)
    energy_kernel, forces_kernel, stress_kernel = model_kernel(batch_padded)

    # Both should produce valid outputs with same shapes
    assert energy_kernel.shape == energy_no_kernel.shape
    assert forces_kernel.shape == forces_no_kernel.shape
    assert stress_kernel.shape == stress_no_kernel.shape

    # Outputs should be finite
    assert np.all(np.isfinite(energy_kernel))
    assert np.all(np.isfinite(forces_kernel))
    assert np.all(np.isfinite(stress_kernel))

    # Outputs should be numerically equivalent
    np.testing.assert_allclose(energy_kernel, energy_no_kernel, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(forces_kernel, forces_no_kernel, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(stress_kernel, stress_no_kernel, rtol=1e-7, atol=1e-7)

    # Test JIT compilation works
    energy_jit, forces_jit, stress_jit = eqx.filter_jit(model_kernel)(batch_padded)
    np.testing.assert_allclose(energy_jit, energy_no_kernel, rtol=1e-7, atol=1e-7)
