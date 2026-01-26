import ase.build
import numpy as np
import pytest

from nequix.calculator import NequixCalculator
from nequix.model import OEQ_AVAILABLE

skip_no_oeq = pytest.mark.skipif(not OEQ_AVAILABLE, reason="OpenEquivariance not installed")


@pytest.mark.parametrize(
    "backend, kernel",
    [
        ("torch", False),
        pytest.param("torch", True, marks=skip_no_oeq),
        ("jax", False),
        pytest.param("jax", True, marks=skip_no_oeq),
    ],
)
def test_calculator_nequix_mp_1(backend, kernel):
    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    calc = NequixCalculator(model_name="nequix-mp-1", backend=backend, use_kernel=kernel)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))


@pytest.mark.parametrize(
    "backend, kernel",
    [
        ("torch", False),
        pytest.param("torch", True, marks=skip_no_oeq),
        ("jax", False),
        pytest.param("jax", True, marks=skip_no_oeq),
    ],
)
def test_calculator_nequix_mp_1_without_cell(backend, kernel):
    atoms = ase.build.molecule("H2O")
    calc = NequixCalculator(model_name="nequix-mp-1", backend=backend, use_kernel=kernel)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
