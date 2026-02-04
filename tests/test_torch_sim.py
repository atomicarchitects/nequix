"""Tests for torch-sim interface."""

import pytest

torch = pytest.importorskip("torch")
ts = pytest.importorskip("torch_sim")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def nequix_model():
    """Load Nequix model for torch-sim."""
    from nequix.torch_sim import NequixTorchSimModel

    return NequixTorchSimModel("nequix-mp-1", device=DEVICE)


@pytest.fixture
def si_state():
    """Create 2-atom Si diamond structure."""
    return ts.SimState(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]], device=DEVICE),
        masses=torch.tensor([28.085, 28.085], device=DEVICE),
        cell=torch.tensor(
            [[[2.72, 2.72, 0.0], [2.72, 0.0, 2.72], [0.0, 2.72, 2.72]]], device=DEVICE
        ),
        pbc=True,
        atomic_numbers=torch.tensor([14, 14], device=DEVICE),
    )


def test_nequix_model_call(nequix_model, si_state):
    """Test that NequixTorchSimModel returns correct output format."""
    result = nequix_model(si_state)

    # Check keys and shapes
    assert result.keys() == {"energy", "forces", "stress"}
    assert result["energy"].shape == (1,)
    assert result["forces"].shape == (2, 3)
    assert result["stress"].shape == (1, 3, 3)

    # Check dtype and device
    for key in result:
        assert result[key].dtype == torch.float64, f"{key} has wrong dtype"
        assert result[key].device.type == si_state.positions.device.type

    # Sanity checks on values
    assert torch.isfinite(result["energy"]).all()
    assert torch.isfinite(result["forces"]).all()
    assert result["forces"].abs().sum() < 100  # Forces should be reasonable magnitude


def test_nequix_model_with_optimizer(nequix_model, si_state):
    """Test that NequixTorchSimModel works with torch-sim optimize."""
    final_state = ts.optimize(
        system=si_state,
        model=nequix_model,
        optimizer=ts.Optimizer.fire,
        max_steps=10,
        convergence_fn=ts.runners.generate_force_convergence_fn(force_tol=0.1),
    )

    assert final_state is not None
    assert final_state.positions.shape == si_state.positions.shape
