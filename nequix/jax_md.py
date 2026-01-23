# Nequix integration with JAX MD

from typing import Any, Callable, Tuple
import jax.numpy as jnp

try:
    from jax_md import custom_partition, partition, space
    from jax_md.custom_partition import Array
except ImportError as e:
    raise ImportError(
        "jax_md is required for the jax_md integration. Install with: pip install git+https://github.com/jax-md/jax-md.git"
    ) from e


def nequix_neighbor_list(
    displacement_fn: Callable,
    box: Array,
    model: Any,
    species: Array,
    neighbor_list_fn: Callable = custom_partition.neighbor_list_multi_image,
    featurizer_fn: Callable = custom_partition.graph_featurizer,
    fractional_coordinates: bool = True,
    **neighbor_kwargs,
) -> Tuple[Any, Callable]:
    """Convenience wrapper to compute nequix energy using a neighbor list.

    Args:
      displacement_fn: Displacement function from ``jax_md.space``.
      box: Box matrix with columns as lattice vectors, shape ``(dim, dim)``.
      model: Pre-trained nequix model with ``cutoff`` and ``node_energies`` attributes.
        Typically obtained from ``nequix.calculator.NequixCalculator(...).model``.
      species: Species indices, shape ``(N,)``.
      neighbor_list_fn: Neighbor list constructor. Defaults to ``custom_partition.neighbor_list_multi_image``
        (uses multi-image neighbor list). Use ``partition.neighbor_list`` for standard MIC.
      featurizer_fn: Function to create a featurizer from ``displacement_fn``. Signature:
        ``featurizer_fn(displacement_fn) -> featurize(species, position, neighbor)``.
        Defaults to ``custom_partition.graph_featurizer`` for multi-image neighbor list.
      fractional_coordinates: Whether positions are in fractional coordinates.
      **neighbor_kwargs: Additional kwargs for neighbor list (e.g., ``max_neighbors``,
        ``dr_threshold``).

    Returns:
      Tuple of ``(neighbor_fn, energy_fn)``:

      - ``neighbor_fn``: Neighbor list with ``allocate`` and ``update`` methods.
      - ``energy_fn(position, neighbor) -> Array``: Computes total energy.
    """
    r_cutoff = model.cutoff

    neighbor_fn = neighbor_list_fn(
        displacement_fn,
        box,
        r_cutoff,
        format=partition.Sparse,
        fractional_coordinates=fractional_coordinates,
        **neighbor_kwargs,
    )

    if featurizer_fn is custom_partition.graph_featurizer:
        displacement_fn, _ = space.free()
        featurizer = featurizer_fn(displacement_fn)
    else:
        featurizer = featurizer_fn(displacement_fn)

    def energy_fn(position: Array, neighbor, **kwargs) -> Array:
        """Compute total energy using the nequix model.

        Args:
          position: Atom positions, shape ``(N, dim)``.
          neighbor: Neighbor list from ``neighbor_fn.allocate`` or ``update``.
          **kwargs: Supports ``perturbation`` for stress via ``quantity.stress``.

        Returns:
          Total energy (scalar) in the model's energy units (typically eV).

        Note:
          For forces, use ``-jax.grad(energy_fn)(position, neighbor)``.
          For stress, use ``quantity.stress(energy_fn, position, box, neighbor=neighbor)``.
        """
        graph = featurizer(species, position, neighbor, **kwargs)
        node_energies = model.node_energies(graph.edges, species, graph.senders, graph.receivers)
        return jnp.sum(node_energies)

    return neighbor_fn, energy_fn
