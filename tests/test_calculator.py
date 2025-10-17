import ase.build
import numpy as np
import pytest

from nequix.calculator import NequixCalculator


@pytest.mark.parametrize("backend, kernel", [("torch", True), ("torch", False), ("jax", False)])
def test_calculator_nequix_mp_1(backend, kernel):
    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    calc = NequixCalculator(model_name="nequix-mp-1", backend=backend, use_kernel=kernel)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)
    print(energy, forces, stress)

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))


@pytest.mark.parametrize("backend, kernel", [("torch", True), ("torch", False), ("jax", False)])
def test_calculator_nequix_mp_1_without_cell(backend, kernel):
    atoms = ase.build.molecule("H2O")
    calc = NequixCalculator(model_name="nequix-mp-1", backend=backend, use_kernel=kernel)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
