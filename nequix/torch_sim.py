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
        """Initialize Nequix model with torch backend."""
        import torch

        from nequix.calculator import NequixCalculator

        self.calculator = NequixCalculator(
            model_name,
            backend="torch",
            use_compile=use_compile,
            use_kernel=use_kernel,
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
        import numpy as np
        import torch
        from ase import Atoms

        # Handle single structure case
        if state.system_idx is None or len(state.system_idx) == 0:
            n_systems = 1
            natoms = torch.tensor([len(state.positions)])
        else:
            n_systems = int(state.system_idx.max().item()) + 1
            natoms = torch.bincount(state.system_idx)

        energies: list[float] = []
        forces_list: list[torch.Tensor] = []
        stresses: list[torch.Tensor] = []
        offset = 0

        for sys_idx in range(n_systems):
            n_atoms = int(natoms[sys_idx].item())
            positions = (
                state.positions[offset : offset + n_atoms].detach().cpu().numpy()
            )
            atomic_numbers = (
                state.atomic_numbers[offset : offset + n_atoms].detach().cpu().numpy()
            )

            # torch-sim uses 'cell' or 'row_vector_cell'
            if hasattr(state, "row_vector_cell") and state.row_vector_cell is not None:
                cell = state.row_vector_cell[sys_idx].detach().cpu().numpy()
            elif hasattr(state, "cell") and state.cell is not None:
                cell = state.cell[sys_idx].detach().cpu().numpy()
            else:
                raise ValueError("State must have cell or row_vector_cell")

            atoms = Atoms(
                numbers=atomic_numbers, positions=positions, cell=cell, pbc=True
            )
            atoms.calc = self.calculator

            energies.append(float(atoms.get_potential_energy()))
            forces_list.append(
                torch.tensor(
                    atoms.get_forces().astype(np.float64),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            stresses.append(
                torch.tensor(
                    atoms.get_stress(voigt=False).astype(np.float64),
                    dtype=self.dtype,
                    device=self.device,
                )
            )
            offset += n_atoms

        return {
            "energy": torch.tensor(energies, dtype=self.dtype, device=self.device),
            "forces": torch.cat(forces_list, dim=0),
            "stress": torch.stack(stresses, dim=0),
        }
