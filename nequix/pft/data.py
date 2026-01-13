import bisect

import jraph
import numpy as np
from nequix.data import AseDBDataset, dict_to_graphstuple, preprocess_graph


# adds a single column of the hessian, and the corresponding vector to
# evaluate the hessian at that column as a global feature
def add_hessian_col_to_graph(graph, hessian_ref, col):
    v1d = (np.arange(hessian_ref.shape[1]) == col).astype(graph.nodes["positions"].dtype)
    vs = v1d.reshape(graph.nodes["positions"].shape)
    hessian_col = hessian_ref[:, col].reshape(graph.nodes["positions"].shape)
    return graph._replace(nodes={**graph.nodes, "vs": vs, "hessian_col": hessian_col})


class PhononDataset(AseDBDataset):
    def __init__(
        self,
        file_path: str,
        atomic_numbers: list[int],
        cutoff: float = 5.0,
        random_col: bool = True,
        seed: int = 42,
        backend: str = "jax",
    ):
        super().__init__(file_path, atomic_numbers, cutoff, backend=backend)
        self.random_col = random_col
        self.rng = np.random.RandomState(seed=seed)
        assert self.backend == "jax", "PhononDataset only supports jax backend for now"

    def _get_graph_dict(self, idx: int):
        db_idx = bisect.bisect(self.id_cumulative, idx)
        if db_idx > 0:
            idx = idx - self.id_cumulative[db_idx - 1]
        row = self.dbs[db_idx]._get_row(self.db_ids[db_idx][idx])
        atoms = row.toatoms()
        # TODO: this should probably be added in the AseDBDataset implementation
        # add data to atoms.info
        if isinstance(row.data, dict):
            atoms.info.update(row.data)
        graph = preprocess_graph(atoms, self.atomic_indices, self.cutoff, True)
        graph["hessian"] = atoms.info["hessian"].astype(np.float32)
        return graph

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        graph_dict = self._get_graph_dict(idx)
        graph = dict_to_graphstuple(graph_dict)
        # use 0 col for validation, otherwise random col
        col = self.rng.randint(0, graph_dict["hessian"].shape[1]) if self.random_col else 0
        return add_hessian_col_to_graph(graph, graph_dict["hessian"], col)
