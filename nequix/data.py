import multiprocessing
from abc import ABC, abstractmethod
import queue
import bisect
import threading
from pathlib import Path

import ase
import ase.db
from ase.geometry import complete_cell
import jax
import jraph
import matscipy.neighbours
import numpy as np
import yaml
from tqdm import tqdm


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: dict[int, int],
    cutoff: float,
    targets: bool,
) -> dict:
    cell = complete_cell(atoms.cell)  # avoids singular cell
    src, dst, shift = matscipy.neighbours.neighbour_list(
        "ijS", positions=atoms.positions, cell=cell, pbc=atoms.pbc, cutoff=cutoff
    )
    graph_dict = {
        "n_node": np.array([len(atoms)]).astype(np.int32),
        "n_edge": np.array([len(src)]).astype(np.int32),
        "senders": dst.astype(np.int32),
        "receivers": src.astype(np.int32),
        "species": np.array([atom_indices[n] for n in atoms.get_atomic_numbers()]).astype(np.int32),
        "positions": atoms.positions.astype(np.float32),
        "shifts": shift.astype(np.float32),
        "cell": atoms.cell.astype(np.float32) if atoms.pbc.all() else None,
    }
    if targets:
        graph_dict["forces"] = atoms.get_forces().astype(np.float32)
        graph_dict["energy"] = np.array([atoms.get_potential_energy()]).astype(np.float32)
        try:
            graph_dict["stress"] = atoms.get_stress(voigt=False).astype(np.float32)
        except ase.calculators.calculator.PropertyNotImplementedError:
            pass

    return graph_dict


def dict_to_pytorch_geometric(graph_dict: dict):
    import torch
    from torch_geometric.data import Data

    """Convert graph dictionary to PyTorch Geometric Data object"""
    # Convert numpy arrays to torch tensors
    species = torch.from_numpy(graph_dict["species"]).long()  # Node features (atomic species)
    positions = torch.from_numpy(graph_dict["positions"])  # Node positions

    # Edge indices (PyG expects [2, num_edges] format)
    edge_index = torch.stack(
        [torch.from_numpy(graph_dict["senders"]), torch.from_numpy(graph_dict["receivers"])], dim=0
    ).long()

    energy = None if "energy" not in graph_dict else torch.from_numpy(graph_dict["energy"])
    forces = None if "forces" not in graph_dict else torch.from_numpy(graph_dict["forces"])
    stress = (
        None if "stress" not in graph_dict else torch.from_numpy(graph_dict["stress"])[None, :, :]
    )

    # Edge attributes
    edge_attr = torch.from_numpy(graph_dict["shifts"])

    cell = (
        torch.from_numpy(graph_dict["cell"])[None, :, :] if graph_dict["cell"] is not None else None
    )

    n_node = torch.from_numpy(graph_dict["n_node"])
    n_edge = torch.from_numpy(graph_dict["n_edge"])

    # Create Data object
    data = Data(
        n_node=n_node,
        n_edge=n_edge,
        energy=energy,
        forces=forces,
        stress=stress,
        x=species,
        positions=positions,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell=cell,
    )

    return data


def dict_to_graphstuple(graph_dict: dict):
    import jraph

    return jraph.GraphsTuple(
        n_node=graph_dict["n_node"],
        n_edge=graph_dict["n_edge"],
        nodes={
            "species": graph_dict["species"],
            "positions": graph_dict["positions"],
            "forces": graph_dict["forces"] if "forces" in graph_dict else None,
        },
        edges={"shifts": graph_dict["shifts"]},
        senders=graph_dict["senders"],
        receivers=graph_dict["receivers"],
        globals={
            "cell": graph_dict["cell"][None, ...] if graph_dict["cell"] is not None else None,
            "energy": graph_dict["energy"] if "energy" in graph_dict else None,
            "stress": graph_dict["stress"][None, ...] if "stress" in graph_dict else None,
        },
    )


def atomic_numbers_to_indices(atomic_numbers: list[int]) -> dict[int, int]:
    """Convert list of atomic numbers to dictionary of atomic number to index."""
    return {n: i for i, n in enumerate(sorted(atomic_numbers))}


class Dataset(ABC):
    def __init__(self, backend: str = "jax"):
        self.backend = backend

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def _get_graph_dict(self, idx: int) -> dict: ...

    def __getitem__(self, idx: int):
        graph = self._get_graph_dict(idx)
        if self.backend == "jax":
            return dict_to_graphstuple(graph)
        if self.backend == "torch":
            return dict_to_pytorch_geometric(graph)
        return graph  # "dict"

    def split(self, valid_frac: float, seed: int = 42):
        n = len(self)
        perm = np.random.RandomState(seed).permutation(n)
        n_tr = int(round(n * (1 - valid_frac)))
        return IndexDataset(self, perm[:n_tr]), IndexDataset(self, perm[n_tr:])


class IndexDataset(Dataset):
    def __init__(self, base: Dataset, indices: np.ndarray):
        super().__init__(backend=base.backend)
        self.base, self.indices = base, np.asarray(indices, dtype=int)

    def __len__(self):
        return self.indices.size

    def _get_graph_dict(self, i: int):
        return self.base._get_graph_dict(int(self.indices[i]))


class ConcatDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        super().__init__(backend=datasets[0].backend)
        self.datasets = datasets
        self.len_cumulative = np.cumsum([len(ds) for ds in datasets])

    def __len__(self):
        return self.len_cumulative[-1]

    def _get_graph_dict(self, idx: int):
        ds_idx = bisect.bisect(self.len_cumulative, idx)
        if ds_idx > 0:
            idx = idx - self.len_cumulative[ds_idx - 1]
        return self.datasets[ds_idx]._get_graph_dict(idx)


# aselmdb dataset for loading omat/omal/odac/salex databases from fairchem
# based on https://github.com/facebookresearch/fairchem/blob/ccc1416/src/fairchem/core/datasets/ase_datasets.py#L382
class AseDBDataset(Dataset):
    def __init__(
        self, file_path: str, atomic_numbers: list[int], cutoff: float = 5.0, backend: str = "jax"
    ):
        super().__init__(backend=backend)
        self.atomic_indices = atomic_numbers_to_indices(atomic_numbers)
        self.file_path = Path(file_path)
        self.cutoff = cutoff
        if self.file_path.is_dir():
            files = sorted(self.file_path.rglob("*.aselmdb"))
            self.dbs = [ase.db.connect(fp, readonly=True, use_lock_file=False) for fp in files]
        else:
            self.dbs = [ase.db.connect(file_path, readonly=True, use_lock_file=False)]
        self.db_ids = [db.ids for db in self.dbs]
        self.id_cumulative = np.cumsum([len(ids) for ids in self.db_ids])

    def __len__(self):
        return self.id_cumulative[-1]

    def _get_graph_dict(self, idx: int):
        db_idx = bisect.bisect(self.id_cumulative, idx)
        if db_idx > 0:
            idx = idx - self.id_cumulative[db_idx - 1]
        atoms = self.dbs[db_idx]._get_row(self.db_ids[db_idx][idx]).toatoms()
        graph = preprocess_graph(atoms, self.atomic_indices, self.cutoff, True)
        return graph


def _dataloader_worker(dataset, index_queue, output_queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break
        output_queue.put((index, dataset[index]))


# multiprocess data loader with dynamic batching, based on
# https://teddykoker.com/2020/12/dataloader/
# https://github.com/google-deepmind/jraph/blob/51f5990/jraph/ogb_examples/data_utils.py
class DataLoader:
    def __init__(
        self,
        dataset,
        max_n_nodes: int,
        max_n_edges: int,
        avg_n_nodes: int,
        avg_n_edges: int,
        batch_size=1,
        n_graph=None,
        seed=0,
        shuffle=False,
        buffer_factor=1.1,
        num_workers=4,
        prefetch_factor=2,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.idxs = np.arange(len(self.dataset))
        self.idx = 0
        self._generator = None  # created in __iter__
        self.n_node = max(batch_size * avg_n_nodes * buffer_factor, max_n_nodes) + 1
        self.n_edge = max(batch_size * avg_n_edges * buffer_factor, max_n_edges)
        self.n_graph = n_graph if n_graph is not None else batch_size + 1
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._started = False
        self.index_queue = None
        self.output_queue = None
        self.workers = []
        self.prefetch_idx = 0

    def _start_workers(self):
        if self._started:
            return

        # NB: we can use fork here, only because we are not using jax
        # in the workers (data is just numpy arrays)
        # multiprocessing.set_start_method("spawn", force=True)
        self._started = True
        self.index_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        for _ in range(self.num_workers):
            worker = multiprocessing.Process(
                target=_dataloader_worker,
                args=(self.dataset, self.index_queue, self.output_queue),
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(seed=hash((self.seed, epoch)) % 2**32)

    def _prefetch(self):
        prefetch_limit = self.idx + self.prefetch_factor * self.num_workers * self.batch_size
        while self.prefetch_idx < len(self.dataset) and self.prefetch_idx < prefetch_limit:
            self.index_queue.put(self.idxs[self.prefetch_idx])
            self.prefetch_idx += 1

    def make_generator(self):
        cache = {}
        self.prefetch_idx = 0

        while True:
            if self.idx >= len(self.dataset):
                return

            self._prefetch()

            real_idx = self.idxs[self.idx]

            if real_idx in cache:
                item = cache[real_idx]
                del cache[real_idx]
            else:
                while True:
                    try:
                        (index, data) = self.output_queue.get(timeout=0)
                    except queue.Empty:
                        continue

                    if index == real_idx:
                        item = data
                        break
                    else:
                        cache[index] = data

            yield item
            self.idx += 1

    def __iter__(self):
        self._start_workers()
        self.idx = 0
        if self.shuffle:
            self.idxs = self.rng.permutation(np.arange(len(self.dataset)))
        self._generator = jraph.dynamically_batch(
            self.make_generator(),
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
        )
        return self

    def __next__(self):
        return next(self._generator)


class ParallelLoader:
    def __init__(self, loader: DataLoader, n: int):
        self.loader = loader
        self.n = n

    def __iter__(self):
        it = iter(self.loader)
        while True:
            try:
                yield jax.tree.map(lambda *x: np.stack(x), *[next(it) for _ in range(self.n)])
            except StopIteration:
                return


# simple threaded prefetching for dataloader (lets us build our dyanamic batches async)
def prefetch(loader, queue_size=4):
    q = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()

    def worker():
        try:
            for item in loader:
                if stop_event.is_set():
                    return
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return
            elif isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        thread.join(timeout=1.0)


# based on https://github.com/ACEsuit/mace/blob/d39cc6b/mace/data/utils.py#L300
def average_atom_energies(dataset: Dataset) -> list[float]:
    """Compute the average energy of each species in the dataset."""
    atomic_indices = dataset.atomic_indices
    A = np.zeros((len(dataset), len(atomic_indices)), dtype=np.float32)
    B = np.zeros((len(dataset),), dtype=np.float32)
    for i, graph in tqdm(enumerate(dataset), total=len(dataset)):
        A[i] = np.bincount(graph.nodes["species"], minlength=len(atomic_indices))
        B[i] = graph.globals["energy"][0]
    E0s = np.linalg.lstsq(A, B, rcond=None)[0].tolist()
    idx_to_atomic_number = {v: k for k, v in atomic_indices.items()}
    atom_energies = {idx_to_atomic_number[i]: e0 for i, e0 in enumerate(E0s)}
    print("computed energies, add to config yml file to avoid recomputing:")
    print(yaml.dump({"atom_energies": atom_energies}))
    return E0s


def dataset_stats(dataset: Dataset, atom_energies: list[float], num_workers: int = 16) -> dict:
    """Compute the statistics of the dataset."""
    atom_energies = np.array(atom_energies)
    num_graphs = len(dataset)

    sum_energy_per_atom = 0.0
    sum_force_sq = 0.0
    num_force_components = 0
    sum_neighbors = 0.0
    sum_nodes = 0
    sum_edges = 0
    max_nodes = 0
    max_edges = 0

    # use DataLoader so we can parallelize workers to compute stats
    loader = DataLoader(
        dataset,
        max_n_nodes=1,
        max_n_edges=1,
        avg_n_nodes=1,
        avg_n_edges=1,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=1,
    )
    loader._start_workers()
    loader.idx = 0
    iterator = loader.make_generator()

    try:
        for graph in tqdm(prefetch(iterator), total=num_graphs):
            n_node = int(np.asarray(graph.n_node).item())
            n_edge = int(np.asarray(graph.n_edge).item())

            sum_nodes += n_node
            sum_edges += n_edge
            if n_node > max_nodes:
                max_nodes = n_node
            if n_edge > max_edges:
                max_edges = n_edge

            graph_e0 = np.sum(atom_energies[graph.nodes["species"]])
            energy_per_atom = float((graph.globals["energy"][0] - graph_e0) / n_node)
            sum_energy_per_atom += energy_per_atom
            sum_force_sq += float(np.sum(graph.nodes["forces"] ** 2))
            num_force_components += graph.nodes["forces"].size
            sum_neighbors += n_edge / n_node
    finally:
        for _ in loader.workers:
            loader.index_queue.put(None)
        for w in loader.workers:
            w.join(timeout=1.0)

    mean = sum_energy_per_atom / num_graphs
    rms = float(np.sqrt(sum_force_sq / num_force_components))
    avg_n_neighbors = sum_neighbors / num_graphs

    stats = {
        "shift": float(mean),
        "scale": float(rms),
        "avg_n_neighbors": float(avg_n_neighbors),
        "avg_n_nodes": float(sum_nodes / num_graphs),
        "avg_n_edges": float(sum_edges / num_graphs),
        "max_n_nodes": int(max_nodes),
        "max_n_edges": int(max_edges),
    }
    print("computed dataset statistics, add to config yml file to avoid recomputing:")
    print(yaml.dump(stats))
    return stats
