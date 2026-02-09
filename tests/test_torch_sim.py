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
    """Test output format: keys, shapes, dtype, device, finite values."""
    result = nequix_model(si_state)

    assert result.keys() == {"energy", "forces", "stress"}
    assert result["energy"].shape == (1,)
    assert result["forces"].shape == (2, 3)
    assert result["stress"].shape == (1, 3, 3)

    for key, val in result.items():
        assert val.dtype == torch.float64, f"{key} wrong dtype"
        assert val.device.type == DEVICE.split(":")[0]
        assert torch.isfinite(val).all(), f"{key} has non-finite values"


def test_nequix_model_with_optimizer(nequix_model, si_state):
    """Test integration with torch-sim optimize."""
    final = ts.optimize(
        system=si_state,
        model=nequix_model,
        optimizer=ts.Optimizer.fire,
        max_steps=10,
        convergence_fn=ts.runners.generate_force_convergence_fn(force_tol=0.1),
    )
    assert final.positions.shape == si_state.positions.shape
