import jraph
import numpy as np

from nequix.data import DataLoader


def _graph() -> jraph.GraphsTuple:
    return jraph.GraphsTuple(
        n_node=np.array([1], dtype=np.int32),
        n_edge=np.array([0], dtype=np.int32),
        nodes=np.zeros((1, 1), dtype=np.float32),
        edges=None,
        senders=np.zeros(0, dtype=np.int32),
        receivers=np.zeros(0, dtype=np.int32),
        globals=None,
    )


def test_dataloader_shutdown():
    loader = DataLoader(
        [_graph()] * 4,
        max_n_nodes=1,
        max_n_edges=0,
        avg_n_nodes=0,
        avg_n_edges=0,
        batch_size=2,
        num_workers=2,
    )

    assert len(list(loader)) == 4
    workers = loader.workers

    loader.shutdown()
    assert not any(w.is_alive() for w in workers)

    # loader is reusable after shutdown
    assert len(list(loader)) == 4
    loader.shutdown()
