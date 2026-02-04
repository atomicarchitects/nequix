"""torch-sim ModelInterface wrapper for NequixCalculator.

This module provides a torch-sim compatible interface for the Nequix model,
enabling batched geometry optimization with torch-sim's autobatching.

Requires: pip install nequix[torch-sim]

Usage:
    import torch_sim as ts
    from nequix.torch_sim import NequixTorchSimModel

    model = NequixTorchSimModel("nequix-oam-1-pft")
    final_state = ts.optimize(
        system=state,
        model=model,
        optimizer=ts.Optimizer.fire,
        max_steps=1000,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch_sim import SimState


class NequixTorchSimModel:
    """torch-sim ModelInterface wrapper for NequixCalculator.

    This wrapper enables using Nequix models with torch-sim's batched optimization
    by converting SimState objects to ASE Atoms for inference.

    Args:
        model_name: Pretrained model alias (e.g., "nequix-oam-1-pft").
        device: Torch device ("cuda" or "cpu").
        use_compile: Enable torch.compile for GPU acceleration.
        use_kernel: Enable OpenEquivariance kernels for GPU acceleration.

    Example:
        >>> model = NequixTorchSimModel("nequix-oam-1-pft")
        >>> state = ts.SimState(...)
        >>> result = model(state)
        >>> print(result["energy"], result["forces"], result["stress"])
    """

    def __init__(
        self,
        model_name: str = "nequix-oam-1-pft",
        device: str = "cuda",
        use_compile: bool = True,
        use_kernel: bool = True,
    ) -> None:
        import torch
        from nequix.calculator import NequixCalculator

        self.calculator = NequixCalculator(
            model_name, backend="torch", use_compile=use_compile, use_kernel=use_kernel
        )
        self.device = torch.device(device)
        self.dtype = torch.float64

    def __call__(self, state: SimState) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given SimState.

        Args:
            state: torch-sim SimState containing positions, cell, and atomic_numbers.

        Returns:
            Dictionary with keys:
                - "energy": Tensor of shape (n_systems,) with total energies in eV
                - "forces": Tensor of shape (n_atoms, 3) with forces in eV/Å
                - "stress": Tensor of shape (n_systems, 3, 3) with stress in eV/Å³
        """
        from ase import Atoms

        import torch

        # Handle single vs batched structures
        if state.system_idx is None or len(state.system_idx) == 0:
            n_systems, natoms = 1, torch.tensor([len(state.positions)])
        else:
            n_systems = int(state.system_idx.max().item()) + 1
            natoms = torch.bincount(state.system_idx)

        energies, forces_list, stresses = [], [], []
        cell_attr = "row_vector_cell" if hasattr(state, "row_vector_cell") else "cell"
        offset = 0

        for idx in range(n_systems):
            n_at = int(natoms[idx].item())
            pos = state.positions[offset : offset + n_at].detach().cpu().numpy()
            nums = state.atomic_numbers[offset : offset + n_at].detach().cpu().numpy()
            cell = getattr(state, cell_attr)[idx].detach().cpu().numpy()

            atoms = Atoms(numbers=nums, positions=pos, cell=cell, pbc=True)
            atoms.calc = self.calculator

            energies.append(float(atoms.get_potential_energy()))
            forces_list.append(
                torch.as_tensor(
                    atoms.get_forces(), dtype=self.dtype, device=self.device
                )
            )
            stresses.append(
                torch.as_tensor(
                    atoms.get_stress(voigt=False), dtype=self.dtype, device=self.device
                )
            )
            offset += n_at

        return {
            "energy": torch.tensor(energies, dtype=self.dtype, device=self.device),
            "forces": torch.cat(forces_list),
            "stress": torch.stack(stresses),
        }
