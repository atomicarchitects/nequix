"""Tests for nequix integration with JAX-MD."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.build import bulk

from nequix.calculator import NequixCalculator
from nequix.data import atomic_numbers_to_indices
from nequix.model import OEQ_AVAILABLE

try:
    from jax_md import quantity, space
    from jax_md.custom_partition import estimate_max_neighbors_from_box

    from nequix.jax_md import nequix_neighbor_list

    JAX_MD_AVAILABLE = True
except ImportError:
    JAX_MD_AVAILABLE = False

pytestmark = pytest.mark.skipif(not JAX_MD_AVAILABLE, reason="jax_md not installed")
skip_no_oeq = pytest.mark.skipif(not OEQ_AVAILABLE, reason="OpenEquivariance not installed")


@pytest.fixture(
    params=[
        pytest.param(False, id="no-kernel"),
        pytest.param(True, id="kernel", marks=skip_no_oeq),
    ]
)
def si_system(request):
    """Create Si diamond system with JAX-MD neighbor list."""
    use_kernel = request.param
    atoms = bulk("Si", "diamond", a=5.43)
    calc = NequixCalculator("nequix-mp-1", use_kernel=use_kernel)
    atoms.calc = calc

    box = jnp.asarray(atoms.cell.T.astype(np.float32))
    atom_indices = atomic_numbers_to_indices(calc.config["atomic_numbers"])
    species = jnp.array([atom_indices[n] for n in atoms.get_atomic_numbers()], dtype=jnp.int32)
    positions = jnp.array(atoms.get_scaled_positions(), dtype=jnp.float32)

    displacement_fn, _ = space.free()
    max_neighbors = estimate_max_neighbors_from_box(box, calc.cutoff, len(atoms), safety_factor=2.0)
    neighbor_fn, energy_fn = nequix_neighbor_list(
        displacement_fn, box, calc.model, species=species, max_neighbors=max_neighbors
    )
    nbrs = neighbor_fn.allocate(positions)

    return atoms, box, species, positions, energy_fn, nbrs


def test_energy_forces_stress(si_system):
    """Test energy, forces, and stress match ASE calculator."""
    atoms, box, species, positions, energy_fn, nbrs = si_system

    # Energy
    E_ase = atoms.get_potential_energy()
    E_jax = float(energy_fn(positions, nbrs))
    np.testing.assert_allclose(E_jax, E_ase, atol=1e-8, rtol=1e-5)

    # Forces
    F_ase = atoms.get_forces()
    F_jax = quantity.force(energy_fn)(positions, nbrs)
    np.testing.assert_allclose(F_jax, F_ase, atol=1e-5, rtol=1e-5)

    # Stress - Note: sign convention w.r.t ASE
    stress_ase = atoms.get_stress(voigt=False)
    stress_jax = -quantity.stress(energy_fn, positions, box, neighbor=nbrs)
    assert stress_jax.shape == (3, 3)
    np.testing.assert_allclose(stress_jax, stress_jax.T, atol=1e-5)
    np.testing.assert_allclose(stress_jax, stress_ase, atol=1e-5, rtol=1e-5)


def test_jit(si_system):
    """Test JIT compilation of energy and force functions."""
    atoms, box, species, positions, energy_fn, nbrs = si_system

    energy_jit = jax.jit(energy_fn)
    force_jit = jax.jit(quantity.force(energy_fn))

    E = energy_jit(positions, nbrs)
    F = force_jit(positions, nbrs)
    jax.block_until_ready(F)

    assert np.isfinite(E)
    assert np.all(np.isfinite(F))


@pytest.mark.parametrize(
    "use_kernel",
    [
        pytest.param(False, id="no-kernel"),
        pytest.param(True, id="kernel", marks=skip_no_oeq),
    ],
)
def test_perturbed_structure(use_kernel):
    """Test energy, forces, and stress on a perturbed structure."""
    atoms = bulk("Si", "diamond", a=5.43)
    calc = NequixCalculator("nequix-mp-1", use_kernel=use_kernel)
    atoms.positions[0] += [0.1, 0.05, -0.05]
    atoms.calc = calc

    box = jnp.asarray(atoms.cell.T.astype(np.float32))
    atom_indices = atomic_numbers_to_indices(calc.config["atomic_numbers"])
    species = jnp.array([atom_indices[n] for n in atoms.get_atomic_numbers()], dtype=jnp.int32)
    positions = jnp.array(atoms.get_scaled_positions(), dtype=jnp.float32)

    displacement_fn, _ = space.free()
    max_neighbors = estimate_max_neighbors_from_box(box, calc.cutoff, len(atoms), safety_factor=2.0)
    neighbor_fn, energy_fn = nequix_neighbor_list(
        displacement_fn, box, calc.model, species=species, max_neighbors=max_neighbors
    )
    nbrs = neighbor_fn.allocate(positions)

    # Energy
    E_ase = atoms.get_potential_energy()
    E_jax = float(energy_fn(positions, nbrs))
    np.testing.assert_allclose(E_jax, E_ase, atol=1e-8, rtol=1e-8)

    # Forces
    F_ase = atoms.get_forces()
    F_jax = quantity.force(energy_fn)(positions, nbrs)
    np.testing.assert_allclose(F_jax, F_ase, atol=1e-5, rtol=1e-5)
    assert np.max(np.abs(F_jax)) > 0.1  # Non-zero forces for perturbed structure

    # Stress - Note: sign convention w.r.t ASE
    stress_ase = atoms.get_stress(voigt=False)
    stress_jax = -quantity.stress(energy_fn, positions, box, neighbor=nbrs)
    assert stress_jax.shape == (3, 3)
    np.testing.assert_allclose(stress_jax, stress_jax.T, atol=1e-5)
    np.testing.assert_allclose(stress_jax, stress_ase, atol=1e-5, rtol=1e-5)
