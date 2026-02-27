import tempfile

import numpy as np
import torch

from nequix.model import load_model as load_model_jax
from nequix.model import save_model as save_model_jax
from nequix.torch_impl.model import load_model as load_model_torch
from nequix.torch_impl.model import save_model as save_model_torch
from nequix.torch_impl.utils import convert_model_jax_to_torch, convert_model_torch_to_jax
from tests.test_model import dummy_graph as dummy_graph_jax
from tests.torch_impl.test_model_torch import dummy_graph as dummy_graph_torch


def test_conversion():
    jax_model, jax_config = load_model_jax("./models/nequix-mp-1.nqx")
    torch_model, torch_config = convert_model_jax_to_torch(jax_model, jax_config, use_kernel=False)

    assert hasattr(torch_model, "atomic_numbers"), "converted model must carry atomic_numbers"
    assert torch_model.atomic_numbers == jax_config["atomic_numbers"]

    tmp_file_torch = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    save_model_torch(tmp_file_torch.name, torch_model, torch_config)

    torch_model_loaded, torch_config_loaded = load_model_torch(tmp_file_torch.name)
    jax_model_converted, jax_config_converted = convert_model_torch_to_jax(
        torch_model_loaded, torch_config_loaded, use_kernel=False
    )

    tmp_file_jax = tempfile.NamedTemporaryFile(suffix=".nqx", delete=False)
    save_model_jax(tmp_file_jax.name, jax_model_converted, jax_config_converted)

    jax_model_loaded, jax_config_loaded = load_model_jax(tmp_file_jax.name)
    torch_model_converted, torch_config_converted = convert_model_jax_to_torch(
        jax_model_converted, jax_config_converted, use_kernel=False
    )

    graph_jax = dummy_graph_jax()

    energy_jax, forces_jax, stress_jax = jax_model(graph_jax)
    energy_converted_jax, forces_converted_jax, stress_converted_jax = jax_model_converted(
        graph_jax
    )

    np.testing.assert_allclose(energy_jax, energy_converted_jax)
    np.testing.assert_allclose(forces_jax, forces_converted_jax)
    np.testing.assert_allclose(stress_jax, stress_converted_jax)

    graph_torch = dummy_graph_torch()
    energy_torch, forces_torch, stress_torch = torch_model_loaded(
        graph_torch.x,
        graph_torch.positions,
        graph_torch.edge_attr,
        graph_torch.edge_index,
        graph_torch.cell,
        graph_torch.n_node,
        graph_torch.n_edge,
        torch.zeros(graph_torch.x.shape[0], dtype=torch.int64),
    )
    energy_converted_torch, forces_converted_torch, stress_converted_torch = torch_model_converted(
        graph_torch.x,
        graph_torch.positions,
        graph_torch.edge_attr,
        graph_torch.edge_index,
        graph_torch.cell,
        graph_torch.n_node,
        graph_torch.n_edge,
        torch.zeros(graph_torch.x.shape[0], dtype=torch.int64),
    )

    np.testing.assert_allclose(
        energy_torch.detach().numpy(), energy_converted_torch.detach().numpy()
    )
    np.testing.assert_allclose(
        forces_torch.detach().numpy(), forces_converted_torch.detach().numpy()
    )
    np.testing.assert_allclose(
        stress_torch.detach().numpy(), stress_converted_torch.detach().numpy()
    )
