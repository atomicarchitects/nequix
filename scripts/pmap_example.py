from pathlib import Path
import time
import functools
import ase
import ase.io
import equinox as eqx
import jraph
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
from nequix.data import atomic_numbers_to_indices, preprocess_graph, dict_to_graphstuple
from nequix.model import Nequix
from nequix.train import loss
from tqdm import tqdm


def train(model, batch, n_epochs=50, lr=0.003, label=None):

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    model = jax.device_put_replicated(model, list(jax.devices()))
    opt_state = jax.device_put_replicated(opt_state, list(jax.devices()))

    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, 0), axis_name="device")
    def train_step(model, opt_state, batch):
        (total_loss, _metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, batch, 1.0, 1.0, 1.0, "mae"
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        updates, opt_state_new = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, total_loss

    device = jax.devices()[0]

    loss_history = []
    step_times = []
    warmup_steps = 5

    for epoch in tqdm(range(n_epochs), desc=label):
        jax.block_until_ready(model)
        step_start = time.perf_counter()

        model, opt_state, total_loss = train_step(model, opt_state, batch)

        jax.block_until_ready(model)
        step_time = time.perf_counter() - step_start

        if epoch >= warmup_steps:
            step_times.append(step_time)

        loss_history.append(float(total_loss.mean().item()))

    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    mem_stats = device.memory_stats()
    peak_mem = mem_stats.get("peak_bytes_in_use", 0) / 1024**3 if mem_stats else 0
    return loss_history, avg_step_time, peak_mem


def example():
    n_devices = len(jax.devices())
    n_epochs = 100
    cutoff = 6.0
    model_kwargs = dict(
        n_species=1,
        cutoff=cutoff,
        hidden_irreps="128x0e + 64x1o + 32x2e + 32x3o",
        n_layers=4,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=2,
    )

    data_path = Path("mp-149.extxyz")
    atoms = ase.io.read(data_path, index="0")
    atomic_indices = atomic_numbers_to_indices(set(atoms.get_atomic_numbers()))
    graph = dict_to_graphstuple(preprocess_graph(atoms, atomic_indices, cutoff, targets=True))
    batch = jraph.pad_with_graphs(graph, n_node=graph.n_node + 1, n_edge=graph.n_edge)

    # replicate batch for each device
    parallel_batch = jax.tree.map(lambda *x: np.stack(x), *[batch for _ in range(n_devices)])

    key = jax.random.key(0)
    model_kernel = Nequix(key=key, kernel=True, **model_kwargs)
    loss_kernel, avg_step_kernel, mem_kernel = train(
        model_kernel, parallel_batch, n_epochs=n_epochs, label="(kernel)"
    )

    key = jax.random.key(0)
    model_no_kernel = Nequix(key=key, kernel=False, **model_kwargs)

    loss_no_kernel, avg_step_no_kernel, mem_no_kernel = train(
        model_no_kernel, parallel_batch, n_epochs=n_epochs, label="(no kernel)"
    )

    steps = np.arange(len(loss_no_kernel))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        steps,
        loss_no_kernel,
        "b-",
        lw=2,
        label=f"No Kernel ({avg_step_no_kernel * 1000:.1f}ms/step, {mem_no_kernel:.2f}GB)",
    )
    ax.plot(
        steps,
        loss_kernel,
        "r--",
        lw=2,
        label=f"Kernel ({avg_step_kernel * 1000:.1f}ms/step, {mem_kernel:.2f}GB)",
    )
    ax.set(xlabel="Step", ylabel=r"Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    example()
