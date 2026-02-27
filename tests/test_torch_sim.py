"""Tests for torch-sim interface."""

import pytest
import torch
from ase.build import bulk

try:
    import torch_sim as ts
    from torch_sim.models.interface import validate_model_outputs
    from torch_sim.testing import (
        SIMSTATE_BULK_GENERATORS,
        assert_model_calculator_consistency,
    )
except (ImportError, OSError, RuntimeError):
    pytest.skip("torch-sim not installed", allow_module_level=True)

from torch_sim.neighbors import torchsim_nl

from nequix.calculator import NequixCalculator
from nequix.torch_sim import NequixTorchSimModel

DEVICE = torch.device("cpu")
DTYPE = torch.float32

# Load raw NequixTorch model at module level (like MACE loads raw_mace_mp)
_calc = NequixCalculator("nequix-mp-1", backend="torch", use_compile=False, use_kernel=False)
raw_nequix_model = _calc.model


@pytest.fixture
def si_atoms():
    """Create Si diamond structure for testing."""
    return bulk("Si", "diamond", a=5.43, cubic=True)


@pytest.fixture
def nequix_calculator():
    """Create Nequix ASE calculator for consistency testing."""
    return NequixCalculator("nequix-mp-1", backend="torch", use_compile=False, use_kernel=False)


@pytest.fixture
def ts_nequix_model():
    """Create NequixTorchSimModel wrapper."""
    return NequixTorchSimModel(
        model=raw_nequix_model,
        device=DEVICE,
        dtype=DTYPE,
        neighbor_list_fn=torchsim_nl,
    )


def test_validate_model_outputs(ts_nequix_model):
    """Test that model conforms to ModelInterface contract."""
    validate_model_outputs(ts_nequix_model, DEVICE, DTYPE)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nequix_dtype_working(si_atoms, dtype):
    """Test that the model works with both float32 and float64."""
    model = NequixTorchSimModel(
        model=raw_nequix_model,
        device=DEVICE,
        dtype=dtype,
        neighbor_list_fn=torchsim_nl,
    )
    state = ts.io.atoms_to_state([si_atoms], DEVICE, dtype)
    model.forward(state)


@pytest.mark.parametrize("sim_state_name", SIMSTATE_BULK_GENERATORS)
def test_nequix_model_consistency(sim_state_name, ts_nequix_model, nequix_calculator):
    """Test consistency between NequixTorchSimModel and NequixCalculator."""
    sim_state = SIMSTATE_BULK_GENERATORS[sim_state_name](DEVICE, DTYPE)
    assert_model_calculator_consistency(
        ts_nequix_model,
        nequix_calculator,
        sim_state,
        force_atol=5e-5,
        stress_atol=5e-5,
    )


def test_nequix_optimize(ts_nequix_model, si_atoms):
    """Test integration with torch-sim optimize."""
    state = ts.io.atoms_to_state([si_atoms], DEVICE, DTYPE)
    final = ts.optimize(
        system=state,
        model=ts_nequix_model,
        optimizer=ts.Optimizer.fire,
        max_steps=10,
        convergence_fn=ts.generate_force_convergence_fn(force_tol=0.1),
    )
    assert final.positions.shape == state.positions.shape
