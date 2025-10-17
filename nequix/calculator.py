import urllib.request
from pathlib import Path

import equinox as eqx
import jraph
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase.geometry import complete_cell

from nequix.data import (
    atomic_numbers_to_indices,
    dict_to_graphstuple,
    dict_to_pytorch_geometric,
    preprocess_graph,
)

file_format_mapper = {
    "jax": "nqx",
    "torch": "pt",
}


class NequixCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    URLS = {
        "nequix-mp-1": "https://figshare.com/files/57245573",
        "nequix-mp-1-kernel": "https://www.dropbox.com/scl/fi/0zavqdbl4n7ep9xxm12lo/nequix-mp-1-kernel.pt?rlkey=7qx8d0pdeo0p3xa74ygkcv655&st=th8f8j12&dl=1",
        "nequix-mp-1-no-kernel": "https://www.dropbox.com/scl/fi/c62lm1b12irf2afezwsud/nequix-mp-1-no-kernel.pt?rlkey=ywpg5qy75e4pco93l1oswmlb6&st=w17qpl2k&dl=1",
    }

    def __init__(
        self,
        model_name: str = "nequix-mp-1",
        model_path: str = None,
        capacity_multiplier: float = 1.1,  # Only for jax backend
        backend: str = "jax",
        use_compile: bool = True,  # Only for torch backend
        use_kernel: bool = True,  # Only for torch backend
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_path is None:
            if backend == "torch":
                import torch

                kernel_name = "kernel" if torch.cuda.is_available() and use_kernel else "no-kernel"
                model_name = f"{model_name}-{kernel_name}"

            filename = f"{model_name}.{file_format_mapper[backend]}"
            for base_path in [Path("./models/"), Path("~/.cache/nequix/models/").expanduser()]:
                model_path = base_path / filename
                if model_path.exists():
                    break

            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(self.URLS[model_name], model_path)

        if backend == "jax":
            from nequix.model import load_model

            self.model, self.config = load_model(model_path)
        elif backend == "torch":
            import torch

            from nequix.torch.model import load_model

            self.model, self.config = load_model(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            # setting compile_state to True would skip compilation else will compile for the first time
            # Only use compile for GPUs
            self.compile_state = False if use_compile and torch.cuda.is_available() else True
        else:
            raise ValueError(f"Backend {backend} not supported")

        self.atom_indices = atomic_numbers_to_indices(self.config["atomic_numbers"])
        self.cutoff = self.config["cutoff"]
        self._capacity = None
        self._capacity_multiplier = capacity_multiplier
        self.backend = backend

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        atoms.cell = complete_cell(atoms.cell)  # avoids singular cell
        processed_graph = preprocess_graph(atoms, self.atom_indices, self.cutoff, False)
        if self.backend == "jax":
            graph = dict_to_graphstuple(processed_graph)
            # maintain edge capacity with _capacity_multiplier over edges,
            # recalculate if numbers (system) changes
            if self._capacity is None or ("numbers" in system_changes):
                self._capacity = int(np.ceil(graph.n_edge[0] * self._capacity_multiplier))
            elif graph.n_edge[0] > self._capacity:
                self._capacity = int(np.ceil(self._capacity * self._capacity_multiplier))
            # Pad the graph
            graph = jraph.pad_with_graphs(
                graph, n_node=graph.n_node[0] + 1, n_edge=self._capacity, n_graph=2
            )
            energy, forces, stress = eqx.filter_jit(self.model)(graph)
            forces = forces[: len(atoms)]

        elif self.backend == "torch":
            import torch

            graph = dict_to_pytorch_geometric(processed_graph)
            graph.n_graph = torch.zeros(graph.x.shape[0], dtype=torch.int64).to(self.device)
            graph = graph.to(self.device)
            if not self.compile_state:
                from torch.fx.experimental.proxy_tensor import make_fx

                self.model = torch.compile(
                    make_fx(
                        self.model,
                        tracing_mode="symbolic",
                        _allow_non_fake_inputs=True,
                        _error_on_data_dependent_ops=True,
                    )(
                        graph.x,
                        graph.positions,
                        graph.edge_attr,
                        graph.edge_index,
                        getattr(graph, "cell", None),
                        graph.n_node,
                        graph.n_edge,
                        graph.n_graph,
                    )
                )
                self.compile_state = True

            # Need to explicitly list out all the tensors because of make_fx
            energy_per_atom, forces, stress = self.model(
                graph.x,
                graph.positions,
                graph.edge_attr,
                graph.edge_index,
                getattr(graph, "cell", None),
                graph.n_node,
                graph.n_edge,
                graph.n_graph,
            )

            # scatter is outside of the model to avoid compile issues
            from nequix.torch.model import scatter

            energy = scatter(energy_per_atom, graph.n_graph, dim=0, dim_size=graph.n_node.size(0))
            energy, forces, stress = (
                energy.detach().cpu(),
                forces.detach().cpu(),
                stress.detach().cpu() if stress is not None else None,
            )

        # take energy and forces without padding
        energy = np.array(energy[0])
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = np.array(forces)
        self.results["stress"] = (
            full_3x3_to_voigt_6_stress(np.array(stress[0])) if stress is not None else None
        )
