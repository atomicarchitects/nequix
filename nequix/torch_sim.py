"""torch-sim ModelInterface wrapper for Nequix (~380 struct/min on H200).

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
    """torch-sim ModelInterface wrapper for Nequix using PyG batching."""

    def __init__(
        self,
        model_name: str = "nequix-oam-1-pft",
        device: str = "cuda",
        use_compile: bool = False,
        use_kernel: bool = True,
    ) -> None:
        """Initialize Nequix model.

        Args:
            model_name: Pretrained model alias (e.g., "nequix-oam-1-pft").
            device: Torch device ("cuda" or "cpu").
            use_compile: Enable torch.compile (default False to avoid dynamo conflicts).
            use_kernel: Enable OpenEquivariance kernels for GPU acceleration.
        """
        import torch

        from nequix.calculator import NequixCalculator

        self.calc = NequixCalculator(
            model_name, backend="torch", use_compile=use_compile, use_kernel=use_kernel
        )
        self.device = torch.device(device)
        self.dtype = torch.float64
        self._memory_scales_with = "n_atoms_x_density"

    def __call__(self, state: SimState) -> dict[str, torch.Tensor]:
        """Compute energies, forces, stresses using PyG batched inference."""
        import torch
        from ase import Atoms
        from torch_geometric.data import Batch

        from nequix.data import dict_to_pytorch_geometric, preprocess_graph
        from nequix.torch.model import scatter

        # Handle single vs batched state
        system_idx = state.system_idx
        if system_idx is None or len(system_idx) == 0:
            system_idx = torch.zeros(
                len(state.positions), dtype=torch.long, device=self.device
            )
        n_systems = int(system_idx.max().item()) + 1

        # Convert to numpy for nequix preprocessing
        positions = state.positions.detach().cpu().numpy()
        cells = state.row_vector_cell.detach().cpu().numpy()
        atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
        sys_idx_np = system_idx.cpu().numpy()

        # Build PyG Data objects using nequix's native preprocessing
        data_list = []
        for idx in range(n_systems):
            mask = sys_idx_np == idx
            atoms = Atoms(
                numbers=atomic_numbers[mask],
                positions=positions[mask],
                cell=cells[idx],
                pbc=True,
            )
            graph = preprocess_graph(
                atoms, self.calc.atom_indices, self.calc.cutoff, targets=False
            )
            data_list.append(dict_to_pytorch_geometric(graph))

        # Batch and move to device
        batch = Batch.from_data_list(data_list)
        dev, f32 = self.device, torch.float32
        species = batch.x.to(dev)
        pos = batch.positions.to(dev, f32)
        edge_attr = batch.edge_attr.to(dev, f32)
        edge_index = batch.edge_index.to(dev)
        cell = batch.cell.to(dev, f32)
        n_node = batch.n_node.to(dev)
        n_edge = batch.n_edge.to(dev)
        n_graph = batch.batch.to(dev)

        # Forward pass
        energy_per_atom, forces, stress = self.calc.model(
            species, pos, edge_attr, edge_index, cell, n_node, n_edge, n_graph
        )

        # Aggregate per-atom energies to per-system
        energies = scatter(energy_per_atom, n_graph, dim=0, dim_size=n_systems)

        return {
            "energy": energies.to(self.dtype),
            "forces": forces.to(self.dtype),
            "stress": stress.to(self.dtype) if stress is not None else None,
        }
