"""torch-sim ModelInterface wrapper for Nequix.

This module provides a TorchSim wrapper of the Nequix model for computing
energies, forces, and stresses for atomistic systems.
"""

from collections.abc import Callable
from pathlib import Path

import torch


from nequix.torch_impl.model import scatter

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
    from torch_sim.typing import StateDict
except ImportError:
    raise ImportError(
        "torch-sim is not installed. Please install it using `pip install torch-sim-atomistic`."
    )


class NequixTorchSimModel(ModelInterface):
    """Computes energies, forces, and stresses using a Nequix model.

    This class wraps a Nequix model for use within the TorchSim framework.
    It supports batched calculations for multiple systems and handles the
    necessary transformations between TorchSim's data structures and
    Nequix's expected inputs.

    Attributes:
        r_max: Cutoff radius for neighbor interactions.
        model: The underlying NequixTorch neural network model.
        neighbor_list_fn: Function used to compute neighbor lists.
    """

    def __init__(
        self,
        model: str | Path | torch.nn.Module,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_fn: Callable = torchsim_nl,
    ) -> None:
        """Initialize the Nequix model for energy and force calculations.

        Args:
            model: Path to a .pt model file, or a pre-loaded NequixTorch module.
            device: Device to run computations on. Defaults to CUDA if available.
            dtype: Data type for tensor operations. Defaults to float64.
            neighbor_list_fn: Function to compute neighbor lists.
                Defaults to torchsim_nl.
        """
        super().__init__()

        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = dtype
        self._compute_stress = True
        self._compute_forces = True
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn
        if isinstance(model, torch.nn.Module):
            config = {
                "cutoff": model.cutoff,
                "atomic_numbers": list(range(model.n_species)),
            }
        else:
            raise TypeError(f"model must be a path or torch.nn.Module, got {type(model)}")

        self.model = model.to(self._device)
        self.model = self.model.eval()

        self.r_max = torch.tensor(config["cutoff"], dtype=self._dtype, device=self._device)

        # GPU lookup table: atomic_number -> species index
        atom_indices = {n: i for i, n in enumerate(sorted(config["atomic_numbers"]))}
        max_z = max(atom_indices.keys())
        z_table = torch.full((max_z + 1,), -1, dtype=torch.long, device=self._device)
        for z, idx in atom_indices.items():
            z_table[z] = idx
        self._z_to_species = z_table

    def forward(self, state: ts.SimState | StateDict) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Args:
            state: SimState or state dict with positions, cell, atomic_numbers,
                system_idx, and pbc fields.

        Returns:
            Dictionary with "energy" [n_systems], "forces" [n_atoms, 3],
            and "stress" [n_systems, 3, 3] tensors.
        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        if sim_state.positions.device != self._device:
            sim_state = sim_state.to(self._device)

        system_idx = sim_state.system_idx
        if system_idx is None or len(system_idx) == 0:
            system_idx = torch.zeros(
                len(sim_state.positions), dtype=torch.long, device=self._device
            )
        n_systems = int(system_idx.max().item()) + 1

        species = self._z_to_species[sim_state.atomic_numbers]

        positions = (
            ts.transforms.pbc_wrap_batched(
                sim_state.positions, sim_state.cell, system_idx, sim_state.pbc
            )
            if sim_state.pbc.any()
            else sim_state.positions
        )

        # Suppress dynamo errors to allow alchemiops @torch.compile to
        # fall back to eager mode when it hits unbacked symbol issues
        # from data-dependent shapes (torch.bincount).
        prev = torch._dynamo.config.suppress_errors
        torch._dynamo.config.suppress_errors = True
        try:
            edge_index, _, unit_shifts = self.neighbor_list_fn(
                positions,
                sim_state.row_vector_cell,
                sim_state.pbc,
                self.r_max,
                system_idx,
            )
        finally:
            torch._dynamo.config.suppress_errors = prev

        energy_per_atom, forces, stress = self.model(
            species,
            positions.to(self._dtype),
            unit_shifts.to(self._dtype),
            edge_index,
            sim_state.row_vector_cell.to(self._dtype),
            None,
            None,
            system_idx,
        )

        energies = scatter(energy_per_atom, system_idx, dim=0, dim_size=n_systems)

        results: dict[str, torch.Tensor] = {
            "energy": energies.to(self._dtype),
        }

        if self._compute_forces:
            results["forces"] = forces.to(self._dtype)

        if self._compute_stress and stress is not None:
            results["stress"] = stress.to(self._dtype)

        return results
