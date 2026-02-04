"""Tests for torch-sim interface."""

import pytest

torch = pytest.importorskip("torch")
ts = pytest.importorskip("torch_sim")


@pytest.fixture
def nequix_model():
    """Load Nequix model for torch-sim."""
    from nequix.torch_sim import NequixTorchSimModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return NequixTorchSimModel("nequix-mp-1", device=device)


@pytest.fixture
def si_state():
    """Create a simple silicon SimState for testing."""
    # 2-atom Si structure
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.36, 1.36, 1.36]], dtype=torch.float64
    )
    cell = torch.tensor(
        [[[2.72, 2.72, 0.0], [2.72, 0.0, 2.72], [0.0, 2.72, 2.72]]], dtype=torch.float64
    )
    atomic_numbers = torch.tensor([14, 14], dtype=torch.int32)
    masses = torch.tensor([28.085, 28.085], dtype=torch.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return ts.SimState(
        positions=positions.to(device),
        masses=masses.to(device),
        cell=cell.to(device),
        pbc=True,
        atomic_numbers=atomic_numbers.to(device),
    )


def test_nequix_model_call(nequix_model, si_state):
    """Test that NequixTorchSimModel returns correct output format."""
    result = nequix_model(si_state)

    assert "energy" in result
    assert "forces" in result
    assert "stress" in result

    assert result["energy"].shape == (1,)
    assert result["forces"].shape == (2, 3)
    assert result["stress"].shape == (1, 3, 3)


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
