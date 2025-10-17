import multiprocessing
import queue
import threading
from pathlib import Path

import ase
import ase.io
import ase.neighborlist
import h5py
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
    src, dst, shift = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
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


def preprocess_file(
    file_path: str, atomic_indices: dict[int, int], cutoff: float
) -> list[dict]:  # Now returns list of dicts
    data = ase.io.read(file_path, index=":", format="extxyz")
    return [preprocess_graph(atoms, atomic_indices, cutoff, True) for atoms in data]


def save_graphs_to_hdf5(graphs, output_path, progress_bar=True):
    """Save graphs to HDF5 file"""
    with h5py.File(output_path, "w") as f:
        f.attrs["n_graphs"] = len(graphs)
        for i, graph_dict in enumerate(
            tqdm(graphs, desc="saving graphs", disable=not progress_bar)
        ):
            grp = f.create_group(f"graph_{i}")
            for key, value in graph_dict.items():
                grp.create_dataset(key, data=value)


def process_worker_files(args):
    """Process files for one worker"""
    worker_id, file_paths, output_path, atomic_indices, cutoff = args
    all_graphs = []
    for file_path in tqdm(
        file_paths,
        desc="reading graphs",
        disable=worker_id != 0,
    ):
        data = ase.io.read(file_path, index=":", format="extxyz")
        graphs = [preprocess_graph(atoms, atomic_indices, cutoff, True) for atoms in data]
        all_graphs.extend(graphs)

    save_graphs_to_hdf5(all_graphs, output_path, progress_bar=worker_id == 0)
    return len(all_graphs)


# pytorch-like dataset that reads xyz files and returns jraph.GraphsTuple
class Dataset:
    def __init__(
        self,
        file_path: str,
        atomic_numbers: list[int],
        cache_dir: str = None,
        split: str = None,
        cutoff: float = 5.0,
        valid_frac: float = 0.1,
        seed: int = 42,
        backend: str = "jax",
    ):
        self.atomic_indices = atomic_numbers_to_indices(atomic_numbers)
        file_path = Path(file_path)
        cache_dir = Path(cache_dir) if cache_dir is not None else file_path.parent
        cache_dir = cache_dir / f"{file_path.stem}_cutoff_{cutoff}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.hdf5_files = sorted(cache_dir.glob("chunk_*.h5"))

        if not self.hdf5_files:
            self._create_cache(file_path, cache_dir, cutoff)
            self.hdf5_files = sorted(cache_dir.glob("chunk_*.h5"))

        self._file_handles = None

        self.index_map = []
        for file_idx, file_handle in enumerate(self.file_handles):
            n_graphs = file_handle.attrs["n_graphs"]
            for local_idx in range(n_graphs):
                self.index_map.append((file_idx, local_idx))

        if split is not None:
            rng = np.random.RandomState(seed=seed)
            perm = rng.permutation(len(self.index_map))
            train_idx, valid_idx = np.split(perm, [int(len(perm) * (1 - valid_frac))])
            indices = train_idx if split == "train" else valid_idx
            self.index_map = [self.index_map[i] for i in indices]

        self.backend = backend

    @property
    def file_handles(self):
        if self._file_handles is None:
            self._file_handles = [h5py.File(hdf5_file, "r") for hdf5_file in self.hdf5_files]
        return self._file_handles

    def __getstate__(self):
        state = self.__dict__.copy()
        # file handles are not picklable
        state["_file_handles"] = None
        return state

    def _create_cache(self, file_path, cache_dir, cutoff):
        if file_path.is_dir():
            file_paths = sorted(file_path.rglob("*.extxyz"))
            n_workers = 16
            chunk_size = len(file_paths) // n_workers + 1
            tasks = []
            for worker_id in range(n_workers):
                start = worker_id * chunk_size
                end = min(start + chunk_size, len(file_paths))
                if start < len(file_paths):
                    worker_files = file_paths[start:end]
                    output_path = cache_dir / f"chunk_{worker_id:04d}.h5"
                    tasks.append(
                        (
                            worker_id,
                            worker_files,
                            output_path,
                            self.atomic_indices,
                            cutoff,
                        )
                    )

            with multiprocessing.Pool(n_workers) as p:
                list(tqdm(p.imap(process_worker_files, tasks), total=len(tasks)))

        else:
            data = ase.io.read(file_path, index=":", format="extxyz")
            graphs = [
                preprocess_graph(atoms, self.atomic_indices, cutoff, True) for atoms in tqdm(data)
            ]
            save_graphs_to_hdf5(graphs, cache_dir / "chunk_0000.h5")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, local_idx = self.index_map[idx]
        grp = self.file_handles[file_idx][f"graph_{local_idx}"]
        graph_dict = {}
        for key in grp:
            graph_dict[key] = grp[key][:]
        if self.backend == "jax":
            return dict_to_graphstuple(graph_dict)
        elif self.backend == "torch":
            return dict_to_pytorch_geometric(graph_dict)

    def __del__(self):
        if hasattr(self, "_file_handles"):
            for fh in self._file_handles:
                fh.close()


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
        self.n_graph = batch_size + 1
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


# def default_collate_fn(graphs):
#     # NB: using batch_np is considerably faster than batch with jax
#     return pad_graph_to_nearest_power_of_two(jraph.batch_np(graphs))


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


def dataset_stats(dataset: Dataset, atom_energies: list[float]) -> dict:
    """Compute the statistics of the dataset."""
    energies, forces, n_neighbors, n_nodes, n_edges = [], [], [], [], []
    atom_energies = np.array(atom_energies)
    for graph in tqdm(dataset, total=len(dataset)):
        graph_e0 = np.sum(atom_energies[graph.nodes["species"]])
        energies.append((graph.globals["energy"][0] - graph_e0) / graph.n_node)
        forces.append(graph.nodes["forces"])
        n_neighbors.append(graph.n_edge / graph.n_node)
        n_nodes.append(graph.n_node)
        n_edges.append(graph.n_edge)
    mean = np.mean(np.concatenate(energies, axis=0))
    rms = np.sqrt(np.mean(np.concatenate(forces, axis=0) ** 2))
    n_neighbors = np.mean(np.concatenate(n_neighbors, axis=0))
    stats = {
        "shift": mean.item(),
        "scale": rms.item(),
        "avg_n_neighbors": n_neighbors.item(),
        "avg_n_nodes": np.mean(n_nodes).item(),
        "avg_n_edges": np.mean(n_edges).item(),
        "max_n_nodes": np.max(n_nodes).item(),
        "max_n_edges": np.max(n_edges).item(),
    }
    print("computed dataset statistics, add to config yml file to avoid recomputing:")
    print(yaml.dump(stats))
    return stats
