import json
import math
import os
from typing import Any, Callable, Optional, Sequence

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from pathlib import Path
import numpy as np

from nequix.layer_norm import RMSLayerNorm

try:
    import torch  # noqa: F401
except ImportError:
    # allow openequivariance to be imported without torch, but only if it is not
    # installed; otherwise, the torch backend won't work if users want to use
    # both torch and jax.
    os.environ["OEQ_NOTORCH"] = "1"

try:
    import openequivariance as oeq
    import openequivariance_extjax  # noqa: F401

    OEQ_AVAILABLE = True
except ImportError:
    OEQ_AVAILABLE = False


def bessel_basis(x: jax.Array, num_basis: int, r_max: float) -> jax.Array:
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,  # prevent division by zero
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: float, p: float) -> jax.Array:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


class NLHRepulsion(eqx.Module):
    CS: np.ndarray = eqx.field(static=True)
    ALPHAS: np.ndarray = eqx.field(static=True)
    atomic_numbers: np.ndarray = eqx.field(static=True)
    ZMAX: int = eqx.field(static=True)
    alchemical_softcore_alpha: float = eqx.field(static=True)
    alchemical_m: float = eqx.field(static=True)

    BOHR_TO_ANG: float = 0.52917721
    HARTREE_TO_EV: float = 27.211386

    def __init__(
        self,
        atomic_numbers: np.ndarray,
        coefficients_file: str = Path(__file__).parent / "nlh_coeffs.dat",
        alchemical_softcore_alpha: float = 0.5,
        alchemical_m: float = 2.0,
    ):
        self.atomic_numbers = np.array(atomic_numbers, dtype=np.int32)

        DATA_NLH = np.loadtxt(coefficients_file, usecols=np.arange(0, 8))
        ZMAX = int(np.max(DATA_NLH[:, 0]))
        AB = np.zeros(((ZMAX + 1) ** 2, 6), dtype=np.float32)
        for i in range(DATA_NLH.shape[0]):
            z1 = int(DATA_NLH[i, 0])
            z2 = int(DATA_NLH[i, 1])
            AB[z1 + ZMAX * z2] = DATA_NLH[i, 2:8]
            AB[z2 + ZMAX * z1] = DATA_NLH[i, 2:8]
        AB = AB.reshape((ZMAX + 1) ** 2, 3, 2)

        self.CS = np.array(AB[:, :, 0])
        self.ALPHAS = np.array(AB[:, :, 1])
        self.ZMAX = ZMAX
        self.alchemical_softcore_alpha = alchemical_softcore_alpha
        self.alchemical_m = alchemical_m

    def __call__(
        self,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        r_norm: jax.Array,
        cutoffs: jax.Array,
        alchemical_group: jax.Array | None = None,
        alchemical_lambda_v: float | None = None,
    ) -> jax.Array:
        Z = jnp.asarray(self.atomic_numbers)[species]
        s12 = Z[senders] + self.ZMAX * Z[receivers]
        cs = jnp.asarray(self.CS)[s12]
        alphas = jnp.asarray(self.ALPHAS)[s12]

        if alchemical_group is not None:
            same_alchemical_group = (
                alchemical_group[senders] == alchemical_group[receivers]
            )
            alch_alpha = self.alchemical_softcore_alpha**2 * (1 - alchemical_lambda_v)
            r_norm = jnp.where(
                same_alchemical_group, r_norm, jnp.sqrt(r_norm**2 + alch_alpha)
            )
            lambda_v = 0.5 * (1 - jnp.cos(jnp.pi * alchemical_lambda_v))
            cutoffs = jnp.where(
                same_alchemical_group, cutoffs, (lambda_v**self.alchemical_m) * cutoffs
            )

        Z = jnp.where(Z > 0, Z.astype(r_norm.dtype), 0.0)
        phi = (cs * jnp.exp(-alphas * r_norm[:, None])).sum(axis=-1)
        Zij = Z[senders] * Z[receivers] * cutoffs
        E_rep_pair = Zij * phi / r_norm
        E_rep = jnp.zeros(Z.shape[0]).at[senders].add(E_rep_pair)
        E_rep *= 0.5 * self.BOHR_TO_ANG * self.HARTREE_TO_EV
        return E_rep


class Sort(eqx.Module):
    irreps: e3nn.Irreps = eqx.field(static=True)
    irreps_sorted: e3nn.Irreps = eqx.field(static=True)
    slices_sorted: list = eqx.field(static=True)

    def __init__(self, irreps: e3nn.Irreps):
        self.irreps = irreps
        slices = list(irreps.slices())
        irreps_sorted, _, inv = irreps.sort()
        self.slices_sorted = [slices[i] for i in inv]
        self.irreps_sorted = irreps_sorted

    def __call__(self, x: jax.Array) -> jax.Array:
        chunks = [x[..., s] for s in self.slices_sorted]
        return jnp.concatenate(chunks, axis=-1)


class Linear(eqx.Module):
    weights: jax.Array
    bias: Optional[jax.Array]
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        key: jax.Array,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = jax.random.normal(key, (in_size, out_size)) * scale
        self.bias = jnp.zeros(out_size) if use_bias else None
        self.use_bias = use_bias

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights)
        if self.use_bias:
            x = x + self.bias
        return x


class MLP(eqx.Module):
    layers: list[Linear]
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        activation=jax.nn.silu,
        *,
        init_scale: float = 1.0,
        use_bias: bool = False,
        key: jax.Array,
    ):
        self.activation = activation

        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [
            Linear(
                sizes[i],
                sizes[i + 1],
                key=keys[i],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
            )
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class NoiseConditionalLinear(eqx.Module):
    """Linear layer whose weights are scaled by a per-sample sigma embedding."""
    linear: e3nn.equinox.Linear
    sigma_mlp: MLP
    irreps_in: e3nn.Irreps = eqx.field(static=True)

    def __init__(self, irreps_in: e3nn.Irreps, irreps_out: e3nn.Irreps, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.irreps_in = irreps_in
        self.linear = e3nn.equinox.Linear(irreps_in=irreps_in, irreps_out=irreps_out, key=k1)
        # outputs one scale per input irrep channel, init to 1
        num_irreps = irreps_in.num_irreps
        self.sigma_mlp = MLP(
            sizes=[1, 64, num_irreps],
            activation=jax.nn.silu,
            use_bias=True,
            key=k2,
        )
        # init last layer to zero weights, one bias -> scales start at 1
        last = self.sigma_mlp.layers[-1]
        self.sigma_mlp = eqx.tree_at(
            lambda m: m.layers[-1].weights, self.sigma_mlp, jnp.zeros_like(last.weights)
        )
        self.sigma_mlp = eqx.tree_at(
            lambda m: m.layers[-1].bias, self.sigma_mlp, jnp.ones_like(last.bias)
        )

    def __call__(self, x: e3nn.IrrepsArray, sigma_per_node: jax.Array) -> e3nn.IrrepsArray:
        # sigma_per_node: (n_nodes,)
        log_sigma = jnp.log(jnp.where(sigma_per_node > 0, sigma_per_node, 1.0))
        scales = jax.vmap(self.sigma_mlp)(log_sigma[:, None])   # (n_nodes, num_irreps)
        scaled_x = e3nn.elementwise_tensor_product(
            x, e3nn.IrrepsArray(f"{self.irreps_in.num_irreps}x0e", scales)
        )
        return self.linear(scaled_x)


class NequixConvolution(eqx.Module):
    output_irreps: e3nn.Irreps = eqx.field(static=True)
    tp_irreps: e3nn.Irreps = eqx.field(static=True)
    index_weights: bool = eqx.field(static=True)
    avg_n_neighbors: float = eqx.field(static=True)
    kernel: bool = eqx.field(static=True)
    tp_conv: Optional[Any] = eqx.field(static=True)

    radial_mlp: MLP
    linear_1: e3nn.equinox.Linear
    linear_2: e3nn.equinox.Linear
    skip: e3nn.equinox.Linear
    layer_norm: Optional[RMSLayerNorm]
    sort: Sort

    def __init__(
        self,
        key: jax.Array,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = True,
        layer_norm: bool = False,
        kernel: bool = False,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors
        self.index_weights = index_weights
        self.kernel = kernel

        irreps_out_tp = []
        instructions = []
        for i, (mul, ir_in1) in enumerate(input_irreps):
            for j, (_, ir_in2) in enumerate(sh_irreps):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in output_irreps:
                        k = len(irreps_out_tp)
                        irreps_out_tp.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        tp_irreps = e3nn.Irreps(irreps_out_tp)
        _, _, inv = tp_irreps.sort()
        self.tp_irreps = tp_irreps

        if kernel:
            instructions = [instructions[i] for i in inv]
            if not OEQ_AVAILABLE:
                raise ImportError(
                    "OpenEquivariance with JAX support is required for kernel=True. "
                    "Install both packages:\n"
                    "  uv pip install 'openequivariance[jax]'\n"
                    "  uv pip install 'openequivariance_extjax' --no-build-isolation"
                )
            problem = oeq.TPProblem(
                str(input_irreps),
                str(sh_irreps),
                str(tp_irreps),
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
            )
            self.tp_conv = oeq.jax.TensorProductConv(problem, deterministic=False)
        else:
            self.tp_conv = None

        self.sort = Sort(tp_irreps)
        tp_irreps = self.sort.irreps_sorted

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.linear_1 = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            key=k1,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            key=k2,
        )

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn.equinox.Linear(
            irreps_in=tp_irreps,
            irreps_out=output_irreps,
            key=k3,
        )

        # skip connection has per-species weights
        self.skip = e3nn.equinox.Linear(
            irreps_in=input_irreps,
            irreps_out=output_irreps,
            linear_type="indexed" if index_weights else "vanilla",
            num_indexed_weights=n_species if index_weights else None,
            force_irreps_out=True,
            key=k4,
        )

        if layer_norm:
            self.layer_norm = RMSLayerNorm(
                irreps=output_irreps,
                centering=False,
                std_balance_degrees=True,
            )
        else:
            self.layer_norm = None

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        messages = self.linear_1(features)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)

        if self.kernel:
            messages_agg = self.sort(
                self.tp_conv.forward(
                    messages.array,
                    sh.array,
                    radial_message,
                    receivers.astype(jnp.int32),
                    senders.astype(jnp.int32),
                )
            )
            messages_agg = e3nn.IrrepsArray(self.sort.irreps_sorted, messages_agg)
        else:
            messages = messages[senders]
            messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.tp_irreps)
            messages = messages * radial_message
            messages_agg = e3nn.scatter_sum(messages, dst=receivers, output_size=features.shape[0])

        messages_agg = messages_agg / jnp.sqrt(jax.lax.stop_gradient(self.avg_n_neighbors))

        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip

        if self.layer_norm is not None:
            features = self.layer_norm(features)

        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )


class Nequix(eqx.Module):
    lmax: int = eqx.field(static=True)
    n_species: int = eqx.field(static=True)
    radial_basis_size: int = eqx.field(static=True)
    radial_polynomial_p: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    shift: float = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    atom_energies: jax.Array
    layers: list[NequixConvolution]
    readout: e3nn.equinox.Linear
    readout_noise: NoiseConditionalLinear
    repulsion_fn: Optional[NLHRepulsion]

    def __init__(
        self,
        key,
        atomic_numbers: list[int],
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e + 128x3o",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        index_weights: bool = True,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
        layer_norm: bool = False,
        kernel: bool = False,
        add_repulsion: bool = False,
    ):
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = len(atomic_numbers)
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.shift = shift
        self.scale = scale
        self.atom_energies = (
            jnp.array(atom_energies)
            if atom_energies is not None
            else jnp.zeros(self.n_species, dtype=jnp.float32)
        )
        input_irreps = e3nn.Irreps(f"{self.n_species}x0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = e3nn.Irreps(hidden_irreps)
        self.layers = []

        key, *subkeys = jax.random.split(key, n_layers + 1)
        for i in range(n_layers):
            self.layers.append(
                NequixConvolution(
                    key=subkeys[i],
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    output_irreps=hidden_irreps if i < n_layers - 1 else hidden_irreps.filter("0e"),
                    sh_irreps=sh_irreps,
                    n_species=self.n_species,
                    radial_basis_size=radial_basis_size,
                    radial_mlp_size=radial_mlp_size,
                    radial_mlp_layers=radial_mlp_layers,
                    mlp_init_scale=mlp_init_scale,
                    avg_n_neighbors=avg_n_neighbors,
                    index_weights=index_weights,
                    layer_norm=layer_norm,
                    kernel=kernel,
                )
            )

        scalar_irreps = hidden_irreps.filter("0e")

        key, readout_key = jax.random.split(key)
        self.readout = e3nn.equinox.Linear(
            irreps_in=scalar_irreps, irreps_out="0e", key=readout_key
        )

        # Noise head: separate weights, same input irreps.
        # Initialised from a fresh key so it starts independent of readout.
        key, noise_key = jax.random.split(key)
        self.readout_noise = NoiseConditionalLinear(
            irreps_in=scalar_irreps, irreps_out="0e", key=noise_key
        )
        self.readout_noise = None

        if add_repulsion:
            print("Adding NLH repulsion term to model.")
            self.repulsion_fn = NLHRepulsion(atomic_numbers=atomic_numbers)
        else:
            self.repulsion_fn = None

    def node_energies(
        self,
        displacements: jax.Array,
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        alchemical_group: Optional[jax.Array] = None,
        alchemical_lambda_e: Optional[float] = None,
        alchemical_lambda_v: Optional[float] = None,
        sigma_per_node: Optional[float] = None,
    ):
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"), jax.nn.one_hot(species, self.n_species)
        )

        square_r_norm = jnp.sum(displacements**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        cutoffs = polynomial_cutoff(r_norm, self.cutoff, self.radial_polynomial_p)

        if alchemical_group is not None:
            same_alchemical_group = (alchemical_group[senders] == alchemical_group[receivers])
            cutoff_scale = (1 - jnp.cos(jnp.pi * alchemical_lambda_e)) / 2
            cutoffs = jnp.where(same_alchemical_group, cutoffs, cutoff_scale * cutoffs)

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.cutoff)
            * cutoffs[:, None]
        )

        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            displacements,
            normalize=True,
            normalization="component",
        )

        for layer in self.layers:
            features = layer(features, species, sh, radial_basis, senders, receivers)

        # base energy
        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * jax.lax.stop_gradient(self.scale) + jax.lax.stop_gradient(
            self.shift
        )

        # add repulsion term if specified
        if self.repulsion_fn is not None:
            node_energies = node_energies + self.repulsion_fn(
                species=species,
                senders=senders,
                receivers=receivers,
                r_norm=r_norm,
                cutoffs=cutoffs,
                alchemical_group=alchemical_group,
                alchemical_lambda_v=alchemical_lambda_v,
            )[:, None]

        # add isolated atom energies to each node as prior
        node_energies = node_energies + jax.lax.stop_gradient(self.atom_energies[species, None])

        # noise conditioning: U(x, sigma) = U(x) + sigma^2 * f(x) 
        if sigma_per_node is not None:
            noise_correction = self.readout_noise(features, sigma_per_node)  # (n_nodes, 1)
            node_energies = node_energies + (sigma_per_node**2)[:, None] * noise_correction

        return node_energies.array

    def __call__(self, data: jraph.GraphsTuple, sigma: Optional[jax.Array] = None):  # <-- sigma_per_node threaded through
        node_graph_index = node_graph_idx(data)
        if sigma is None:
            sigma_per_node = None
        else:
            sigma_per_node = sigma[node_graph_index]

        if data.globals["cell"] is None:
            def total_energy_fn(positions: jax.Array):
                r = positions[data.senders] - positions[data.receivers]
                node_energies = self.node_energies(
                    r, data.nodes["species"], data.senders, data.receivers,
                    sigma_per_node=sigma_per_node,
                )
                return jnp.sum(node_energies), node_energies

            minus_forces, node_energies = eqx.filter_grad(total_energy_fn, has_aux=True)(
                data.nodes["positions"]
            )
        else:
            def total_energy_fn(positions_eps: tuple[jax.Array, jax.Array]):
                positions, eps = positions_eps
                eps_sym = (eps + eps.swapaxes(1, 2)) / 2
                eps_sym_per_node = jnp.repeat(
                    eps_sym, data.n_node, axis=0,
                    total_repeat_length=data.nodes["positions"].shape[0],
                )
                positions = positions + jnp.einsum("ik,ikj->ij", positions, eps_sym_per_node)
                cell = data.globals["cell"] + jnp.einsum(
                    "bij,bjk->bik", data.globals["cell"], eps_sym
                )
                cell_per_edge = jnp.repeat(
                    cell, data.n_edge, axis=0,
                    total_repeat_length=data.edges["shifts"].shape[0],
                )
                offsets = jnp.einsum("ij,ijk->ik", data.edges["shifts"], cell_per_edge)
                r = positions[data.senders] - positions[data.receivers] + offsets
                node_energies = self.node_energies(
                    r, data.nodes["species"], data.senders, data.receivers,
                    sigma_per_node=sigma_per_node,
                )
                return jnp.sum(node_energies), node_energies

            eps = jnp.zeros_like(data.globals["cell"])
            (minus_forces, virial), node_energies = eqx.filter_grad(total_energy_fn, has_aux=True)(
                (data.nodes["positions"], eps)
            )

        # padded nodes may have nan forces, so we mask them
        node_mask = jraph.get_node_padding_mask(data)
        minus_forces = jnp.where(node_mask[:, None], minus_forces, 0.0)

        # compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        if data.globals["cell"] is None:
            stress = None
        else:
            det = jnp.abs(jnp.linalg.det(data.globals["cell"]))[:, None, None]
            det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
            stress = virial / det
            # padded stress may be nan, so we mask them
            graph_mask = jraph.get_graph_padding_mask(data)
            stress = jnp.where(graph_mask[:, None, None], stress, 0.0)

        return graph_energies[:, 0], -minus_forces, stress


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    # based on https://github.com/google-deepmind/jraph/blob/51f5990/jraph/_src/models.py#L209-L216
    n_graph = data.n_node.shape[0]
    # equivalent to jnp.sum(n_node), but jittable
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node)
    return node_gr_idx


def weight_decay_mask(model):
    """Returns a pytree with the same structure as the model, where each leaf is a boolean indicating whether to apply weight decay to that leaf."""

    def is_layer(x):
        return isinstance(x, (Linear, e3nn.equinox.Linear))

    def set_mask(x):
        if isinstance(x, Linear):
            mask = jax.tree.map(lambda _: True, x)
            mask = eqx.tree_at(lambda m: m.bias, mask, False)
            return mask
        elif isinstance(x, e3nn.equinox.Linear):
            return jax.tree.map(lambda _: True, x)
        else:
            return jax.tree.map(lambda _: False, x)

    mask = jax.tree.map(set_mask, model, is_leaf=is_layer)
    return mask


def save_model(path: str, model: eqx.Module, config: dict):
    """Save a model and its config to a file."""
    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str, kernel: bool = False) -> tuple[Nequix, dict]:
    """Load a model and its config from a file."""
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = Nequix(
            key=jax.random.key(0),
            atomic_numbers=config["atomic_numbers"],
            hidden_irreps=config["hidden_irreps"],
            lmax=config["lmax"],
            cutoff=config["cutoff"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            index_weights=config["index_weights"],
            layer_norm=config["layer_norm"],
            shift=config["shift"],
            scale=config["scale"],
            avg_n_neighbors=config["avg_n_neighbors"],
            kernel=kernel,
            add_repulsion=config.get("add_repulsion", False),
            # NOTE: atom_energies will be in model weights
        )
        model = eqx.tree_deserialise_leaves(f, model)
        return model, config
