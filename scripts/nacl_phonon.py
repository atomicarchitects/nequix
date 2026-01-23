"""Benchmark phonon spectrum calculation with Phonopy: ASE vs JAX-MD, kernel vs no-kernel."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import bulk
from jax_md import quantity, space
from jax_md.custom_partition import estimate_max_neighbors_from_box
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from nequix.calculator import NequixCalculator
from nequix.data import atomic_numbers_to_indices
from nequix.jax_md import nequix_neighbor_list

# Band path for NaCl (FCC Brillouin zone)
BAND_PATH = [
    [0, 0, 0],
    [0.5, 0, 0.5],
    [0.5, 0.25, 0.75],
    [0.5, 0.5, 0.5],
    [0, 0, 0],
    [0.375, 0.375, 0.75],
]
BAND_LABELS = ["$\\Gamma$", "X", "W", "L", "$\\Gamma$", "K"]
N_SUPERCELL, NPOINTS = 3, 51


def setup_phonopy(atoms, supercell_matrix):
    ph_atoms = PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),
    )
    phonon = Phonopy(ph_atoms, supercell_matrix)
    phonon.generate_displacements(distance=0.01)
    return phonon


def run_band_structure(phonon, forces):
    phonon.forces = forces
    phonon.produce_force_constants()
    paths = [
        np.linspace(BAND_PATH[i], BAND_PATH[i + 1], NPOINTS) for i in range(len(BAND_PATH) - 1)
    ]
    phonon.run_band_structure(paths, with_eigenvectors=False, labels=BAND_LABELS)
    band = phonon.get_band_structure_dict()
    return band["frequencies"], band["distances"]


def compute_ase(atoms, use_kernel, supercell_matrix):
    calc = NequixCalculator("nequix-mp-1", use_kernel=use_kernel)
    phonon = setup_phonopy(atoms, supercell_matrix)
    forces = []
    for sc in phonon.supercells_with_displacements:
        sc_ase = Atoms(
            symbols=sc.symbols, cell=sc.cell, scaled_positions=sc.scaled_positions, pbc=True
        )
        sc_ase.calc = calc
        forces.append(sc_ase.get_forces())
    freq, dist = run_band_structure(phonon, forces)
    return {"frequencies": freq, "distances": dist}


def compute_jax_md(atoms, use_kernel, supercell_matrix):
    calc = NequixCalculator("nequix-mp-1", use_kernel=use_kernel)
    phonon = setup_phonopy(atoms, supercell_matrix)
    supercells = phonon.supercells_with_displacements
    sc0 = supercells[0]

    box = jnp.array(sc0.cell.T, dtype=jnp.float32)
    atom_idx = atomic_numbers_to_indices(calc.config["atomic_numbers"])
    species = jnp.array([atom_idx[n] for n in sc0.numbers], dtype=jnp.int32)
    displacement_fn, _ = space.periodic_general(box, fractional_coordinates=True)
    max_nbrs = estimate_max_neighbors_from_box(
        box, calc.cutoff, len(sc0.numbers), safety_factor=2.0
    )
    neighbor_fn, energy_fn = nequix_neighbor_list(
        displacement_fn, box, calc.model, species=species, max_neighbors=max_nbrs
    )
    force_fn = jax.jit(quantity.force(energy_fn))
    nbrs = neighbor_fn.allocate(jnp.array(sc0.scaled_positions, dtype=jnp.float32))

    forces = []
    for sc in supercells:
        pos = jnp.array(sc.scaled_positions, dtype=jnp.float32)
        forces.append(np.array(force_fn(pos, nbrs)))
    freq, dist = run_band_structure(phonon, forces)
    return {"frequencies": freq, "distances": dist}


def plot_overlay(ax, res_no_kernel, res_kernel, title):
    for i, (dist, freq) in enumerate(zip(res_no_kernel["distances"], res_no_kernel["frequencies"])):
        for j in range(freq.shape[1]):
            ax.plot(
                dist,
                freq[:, j],
                "C0",
                lw=6,
                alpha=0.5,
                label="no kernel" if i == 0 and j == 0 else None,
            )
    for i, (dist, freq) in enumerate(zip(res_kernel["distances"], res_kernel["frequencies"])):
        for j in range(freq.shape[1]):
            ax.plot(dist, freq[:, j], "k", lw=1, label="kernel" if i == 0 and j == 0 else None)

    ticks = [res_no_kernel["distances"][0][0]] + [d[-1] for d in res_no_kernel["distances"]]
    for x in ticks[1:-1]:
        ax.axvline(x, c="k", lw=0.5, ls="--")
    ax.set(xticks=ticks, xlim=(ticks[0], ticks[-1]), ylabel="Frequency (THz)")
    ax.set_xticklabels(BAND_LABELS, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.axhline(0, c="k", lw=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=14, frameon=False)


if __name__ == "__main__":
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    supercell_matrix = np.diag([N_SUPERCELL] * 3)

    print("Computing phonon spectra...")
    results = {
        "ASE (no kernel)": compute_ase(atoms, False, supercell_matrix),
        "ASE (kernel)": compute_ase(atoms, True, supercell_matrix),
        "JAX-MD (no kernel)": compute_jax_md(atoms, False, supercell_matrix),
        "JAX-MD (kernel)": compute_jax_md(atoms, True, supercell_matrix),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_overlay(axes[0], results["ASE (no kernel)"], results["ASE (kernel)"], "ASE")
    plot_overlay(axes[1], results["JAX-MD (no kernel)"], results["JAX-MD (kernel)"], "JAX-MD")
    fig.suptitle("NaCl Phonon Spectrum", fontsize=14)
    plt.tight_layout()
    plt.savefig("phonon_spectrum.png", dpi=150)
