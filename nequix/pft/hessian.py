import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def hessian_linearized(model, graph, batch_size=None):
    cell_per_edge = jnp.repeat(
        graph.globals["cell"],
        graph.n_edge,
        axis=0,
        total_repeat_length=graph.edges["shifts"].shape[0],
    )
    offsets = jnp.einsum("ij,ijk->ik", graph.edges["shifts"], cell_per_edge)

    def total_energy_fn(positions):
        r = positions[graph.senders] - positions[graph.receivers] + offsets
        node_energies = model.node_energies(
            r, graph.nodes["species"], graph.senders, graph.receivers
        )
        return jnp.sum(node_energies)

    pos = graph.nodes["positions"]
    _, hvp = jax.linearize(jax.grad(total_energy_fn), pos)
    hvp = jax.jit(hvp)
    basis = jnp.eye(pos.shape[0] * pos.shape[1]).reshape(-1, *pos.shape)
    return (
        jax.lax.map(hvp, basis, batch_size=batch_size)
        .reshape(pos.shape + pos.shape)  # (n, 3, n, 3)
        .swapaxes(1, 2)  # (n, n, 3, 3)
    )
