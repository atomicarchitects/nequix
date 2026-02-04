import urllib.request
from pathlib import Path

import equinox as eqx
import jraph
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from nequix.data import (
    atomic_numbers_to_indices,
    dict_to_graphstuple,
    dict_to_pytorch_geometric,
    preprocess_graph,
)


from nequix.model import load_model as load_model_jax
from nequix.model import save_model as save_model_jax


class NequixCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    URLS = {
        # NOTE: not using figshare urls because they like to give 403 errors
        # "nequix-mp-1": "https://figshare.com/files/57245573",
        # "nequix-mp-1-pft": "https://figshare.com/files/60965527",
        # "nequix-mp-1-pft-no-cotrain": "https://figshare.com/files/60965530",
        #
        # MP models
        "nequix-mp-1": "https://github.com/atomicarchitects/nequix/raw/7c2854de8e754b1a60274c7d9d2e014989ed632e/models/nequix-mp-1.nqx",
        "nequix-mp-1-pft": "https://github.com/atomicarchitects/nequix/raw/7c2854de8e754b1a60274c7d9d2e014989ed632e/models/nequix-mp-1-pft.nqx",
        "nequix-mp-1-pft-no-cotrain": "https://github.com/atomicarchitects/nequix/raw/7c2854de8e754b1a60274c7d9d2e014989ed632e/models/nequix-mp-1-pft-no-cotrain.nqx",
        #
        # OMat models
        "nequix-omat-1": "https://github.com/atomicarchitects/nequix/raw/9c852a9c4cce69ae5e75bc5a2cf670be21ff8c28/models/nequix-omat-1.nqx",
        "nequix-oam-1": "https://github.com/atomicarchitects/nequix/raw/9c852a9c4cce69ae5e75bc5a2cf670be21ff8c28/models/nequix-oam-1.nqx",
        "nequix-oam-1-pft": "https://github.com/atomicarchitects/nequix/raw/9c852a9c4cce69ae5e75bc5a2cf670be21ff8c28/models/nequix-oam-1-pft.nqx",
    }

    def __init__(
        self,
        model_name: str = "nequix-mp-1",
        model_path: str = None,
        capacity_multiplier: float = 1.1,  # Only for jax backend
        backend: str = "jax",
        use_kernel: bool = True,  # Only for torch backend
        use_compile: bool = True,  # Only for torch backend
        **kwargs,
    ):
        super().__init__(**kwargs)

        if use_kernel and backend == "torch":
            import torch

            assert torch.cuda.is_available(), "Kernels need GPU environment"

        base_path = Path("~/.cache/nequix/models/").expanduser()
        ext_for_backend = "nqx" if backend == "jax" else "pt"

        if model_path is not None:
            model_path = Path(model_path)
        else:
            # attempt to load checkpoint with desired backend
            model_path = base_path / f"{model_name}.{ext_for_backend}"
            if not model_path.exists():
                # otherwise use nqx checkpoint
                model_path = base_path / f"{model_name}.nqx"
                if not model_path.exists():
                    # download if necessary
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(self.URLS[model_name], model_path)

        path_backend = "jax" if model_path.suffix == ".nqx" else "torch"
        if path_backend == backend:
            if backend == "jax":
                self.model, self.config = load_model_jax(model_path)
            else:
                from nequix.torch.model import load_model as load_model_torch

                self.model, self.config = load_model_torch(model_path, use_kernel)
        else:
            # Convert and save
            if path_backend == "torch":
                from nequix.torch.utils import convert_model_torch_to_jax

                torch_model, torch_config = load_model_torch(model_path, use_kernel)
                print("Converting PyTorch model to JAX ...")
                self.model, self.config = convert_model_torch_to_jax(torch_model, torch_config)
                out_path = model_path.parent / f"{model_name}.nqx"
                save_model_jax(out_path, self.model, self.config)
            else:
                from nequix.torch.utils import convert_model_jax_to_torch

                jax_model, jax_config = load_model_jax(model_path)
                print("Converting JAX model to PyTorch ...")
                self.model, self.config = convert_model_jax_to_torch(
                    jax_model, jax_config, use_kernel
                )
                out_path = model_path.parent / f"{model_name}.pt"
                from nequix.torch.model import save_model as save_model_torch

                save_model_torch(out_path, self.model, self.config)
            print("Model saved to ", out_path)

        if backend == "torch":
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            # setting compile_state to True would skip compilation else will compile for the first time
            # Only use compile for GPUs
            self.compile_state = False if use_compile and torch.cuda.is_available() else True

        self.atom_indices = atomic_numbers_to_indices(self.config["atomic_numbers"])
        self.cutoff = self.config["cutoff"]
        self._capacity = None
        self._capacity_multiplier = capacity_multiplier
        self.backend = backend

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms)
        processed_graph = preprocess_graph(atoms, self.atom_indices, self.cutoff, False)
        if self.backend == "jax":
            graph = dict_to_graphstuple(processed_graph)
            # maintain edge capacity with _capacity_multiplier over edges,
            # recalculate if numbers (system) changes, or if the capacity is exceeded
            if (
                self._capacity is None
                or ("numbers" in system_changes)
                or graph.n_edge[0] > self._capacity
            ):
                raw = int(np.ceil(graph.n_edge[0] * self._capacity_multiplier))
                # round up edges to the nearest multiple of 64
                # NB: this avoids excessive recompilation in high-throughput
                # workflows (e.g.  material relaxtions) but this number may need
                # to be tuned depending on the system sizes
                self._capacity = ((raw + 63) // 64) * 64

            # round up nodes to the nearest multiple of 8
            # NB: this avoids excessive recompilation in high-throughput
            # workflows (e.g. material relaxtions) but this number may need to
            # be tuned depending on the system sizes
            n_node = ((graph.n_node[0] + 8) // 8) * 8

            # pad the graph
            graph = jraph.pad_with_graphs(graph, n_node=n_node, n_edge=self._capacity, n_graph=2)
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
