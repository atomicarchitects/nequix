import json
import math
from typing import Callable, List, Mapping, Optional, Sequence, Union

import numpy as np
import scipy
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode

from nequix.torch.layer_norm import RMSLayerNorm

# Borrowed from https://github.com/mir-group/pytorch_runstats


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(
            size,
            dtype=(torch.float32 if src.dtype not in (torch.float32, torch.float64) else src.dtype),
            device=src.device,
        )
        return out.scatter_add_(dim, index, src.to(out.dtype))
    else:
        return out.scatter_add_(dim, index, src)


def bessel_basis(x: torch.Tensor, num_basis: int, r_max: float) -> torch.Tensor:
    prefactor = 2.0 / r_max
    bessel_weights = torch.linspace(1.0, num_basis, num_basis, device=x.device) * torch.pi
    x = x[:, None]
    return prefactor * torch.where(
        x == 0.0,
        bessel_weights / r_max,  # prevent division by zero
        torch.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: torch.Tensor, r_max: float, p: float) -> torch.Tensor:
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
    out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
    return out * torch.where(x < 1.0, 1.0, 0.0)


# Borrowed from https://github.com/e3nn/e3nn.git


def normalspace(n: int, device=None, dtype=None) -> torch.Tensor:
    return np.sqrt(2) * scipy.special.erfinv(np.linspace(-1.0, 1.0, n + 2)[1:-1])


@compile_mode("trace")
class normalize2mom(torch.nn.Module):
    _is_id: bool
    cst: float

    def __init__(
        # pylint: disable=unused-argument
        self,
        f,
        dtype=None,
        device=None,
    ) -> None:
        super().__init__()

        # Try to infer a device:
        if device is None and isinstance(f, torch.nn.Module):
            # Avoid circular import
            from e3nn.util._argtools import _get_device

            device = _get_device(f)

        with torch.no_grad():
            x = torch.from_numpy(normalspace(1_000_001))
            c = torch.mean(f(x) ** 2) ** 0.5
            c = c.item()

            if np.allclose(c, 1.0):
                self.f = f
            else:
                self.f = lambda x: f(x) / c

        self.x = x

    def forward(self, x):
        return self.f(x)


def parity_function(phi: Callable[[float], float]) -> int:
    x = np.linspace(0.0, 10.0, 256)

    a1, a2 = phi(x), phi(-x)
    if np.max(np.abs(a1 - a2)) < 1e-5:
        return 1
    elif np.max(np.abs(a1 + a2)) < 1e-5:
        return -1
    else:
        return 0


def soft_odd(x):
    return (1 - torch.exp(-(x**2))) * x


@compile_mode("trace")
class Activation(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        acts: List[Optional[Callable[[float], float]]] = None,
        *,
        even_act: Callable[[float], float] = torch.nn.functional.gelu,
        odd_act: Callable[[float], float] = soft_odd,
        normalize_act: bool = True,
    ) -> None:
        super().__init__()
        irreps_in = o3.Irreps(irreps_in)

        if acts is None:
            acts = [{1: even_act, -1: odd_act}[ir.p] if ir.l == 0 else None for _, ir in irreps_in]

        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        if normalize_act:
            acts = [normalize2mom(act) if act is not None else None for act in acts]

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError(
                        "Activation: cannot apply an activation function to a non-scalar input."
                    )

                p_out = parity_function(act) if p_in == -1 else p_in

                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError(
                        "Activation: the parity is violated! The input scalar is odd but the activation is neither "
                        "even nor odd."
                    )
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)
        self.acts = torch.nn.ModuleList(acts)
        self.paths = [(mul, (l, p), act) for (mul, (l, p)), act in zip(self.irreps_in, self.acts)]  # noqa: E741

    def forward(self, features, dim: int = -1):
        output = []
        index = 0
        for mul, (l, _), act in self.paths:  # noqa: E741
            ir_dim = 2 * l + 1
            if act is not None:
                x = features.narrow(dim, index, mul)
                if x is None:
                    if torch.allclose(act(torch.tensor(0.0)), 0.0):
                        output.append(x)
                    else:
                        output.append(
                            act(
                                torch.ones(features.shape[:-1] + (mul, 1)).to(
                                    dtype=features.dtype, device=features.device
                                )
                            )
                        )
                else:
                    output.append(act(x))
            else:
                x = features.narrow(dim, index, mul * ir_dim)
                output.append(x)
            index += mul * ir_dim

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)


class Gate(torch.nn.Module):
    def __init__(
        self,
        irreps_out: Union[str, o3.Irreps],
        act: Optional[Mapping[int, torch.nn.Module]] = None,
        act_gates: Optional[Mapping[int, torch.nn.Module]] = None,
    ):
        # Note: Does not work with odd scalar inputs
        super().__init__()

        self.irreps_out = o3.Irreps(irreps_out)

        if act is None:
            act = {
                1: torch.nn.SiLU(),
                -1: torch.nn.Tanh(),
            }

        if act_gates is None:
            act_gates = {
                1: torch.nn.SiLU(),
                -1: torch.nn.Tanh(),
            }

        scalars = self.irreps_out.filter(keep=["0e", "0o"])
        vectors = self.irreps_out.filter(drop=["0e", "0o"])

        scalars_extra = scalars.slice_by_mul[: scalars.dim - vectors.num_irreps]
        scalars_gates = scalars.slice_by_mul[scalars.dim - vectors.num_irreps :]
        self.scalars_gates_vectors_elemwise = o3.ElementwiseTensorProduct(scalars_gates, vectors)
        self.slice_scalars_extra = slice(0, scalars.dim - vectors.num_irreps)
        self.slice_scalars_gates = slice(scalars.dim - vectors.num_irreps, irreps_out.dim)
        self.slice_vectors = slice(scalars.dim, self.irreps_out.dim)

        self.scalars_extra = Activation(scalars_extra, [act[ir.p] for _, ir in scalars_extra])
        self.scalars_gates = Activation(scalars_gates, [act_gates[ir.p] for _, ir in scalars_gates])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gate to the input tensor."""
        scalars_extra = x[:, self.slice_scalars_extra]
        scalars_gates = x[:, self.slice_scalars_gates]
        vectors = x[:, self.slice_vectors]
        return torch.concatenate(
            [
                self.scalars_extra(scalars_extra),
                self.scalars_gates_vectors_elemwise(self.scalars_gates(scalars_gates), vectors),
            ],
            dim=-1,
        )


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.use_bias = use_bias
        scale = math.sqrt(init_scale / in_size)

        self.weights = torch.nn.Parameter(torch.randn(in_size, out_size) * scale)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_size))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weights
        if self.use_bias and self.bias is not None:
            x = x + self.bias
        return x


class MLP(torch.nn.Module):
    def __init__(
        self,
        sizes,
        activation=torch.nn.SiLU,
        *,
        init_scale: float = 1.0,
        use_bias: bool = True,
    ):
        super().__init__()
        self.activation = activation()

        self.layers = torch.nn.ModuleList(
            [
                Linear(
                    sizes[i],
                    sizes[i + 1],
                    use_bias=use_bias,
                    # don't scale last layer since no activation
                    init_scale=init_scale if i < len(sizes) - 2 else 1.0,
                )
                for i in range(len(sizes) - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


class Sort(torch.nn.Module):
    "Sorts the irreps and uses it to slice the Tensor"

    def __init__(self, irreps: o3.Irreps):
        super().__init__()
        self.irreps = irreps
        slices = []
        for slice_ir in irreps.slices():
            slices.append(slice_ir)

        irreps_sorted, _, inv = irreps.sort()
        self.slices_sorted = [slices[i] for i in inv]
        self.irreps_sorted = irreps_sorted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for slice_ir in self.slices_sorted:
            chunks.append(x[:, slice_ir])
        return torch.cat(chunks, dim=-1)


class NequixTorchConvolution(torch.nn.Module):
    output_irreps: o3.Irreps
    index_weights: bool = False

    radial_mlp: MLP
    linear_1: o3.Linear
    linear_2: o3.Linear
    skip: o3.Linear
    layer_norm: Optional[RMSLayerNorm]

    def __init__(
        self,
        input_irreps: o3.Irreps,
        output_irreps: o3.Irreps,
        sh_irreps: o3.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        index_weights: bool = False,
        layer_norm: bool = False,
        kernel: bool = False,
    ):
        super().__init__()

        self.output_irreps = output_irreps
        # TODO: This is not supported in o3.Linear
        assert not index_weights, "Index weights are not supported in o3.Linear"
        self.index_weights = index_weights
        self.kernel = kernel
        self.register_buffer("avg_n_neighbors", torch.tensor(avg_n_neighbors))

        self.linear_1 = o3.Linear(
            irreps_in=input_irreps.regroup(),
            irreps_out=input_irreps.regroup(),
        )

        # Separable Tensor Product
        irreps_out_tp = []
        instructions_tp = []
        for i, (mul, ir_in1) in enumerate(input_irreps):
            for j, (_, ir_in2) in enumerate(sh_irreps):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in output_irreps or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_out_tp)
                        irreps_out_tp.append((mul, ir_out))
                        instructions_tp.append((i, j, k, "uvu", True))

        tp_irreps = o3.Irreps(irreps_out_tp)
        _, _, inv = tp_irreps.sort()  # sort irreps
        instructions_tp = [instructions_tp[i] for i in inv]  # sort instructions

        if self.kernel:
            from openequivariance import TensorProductConv, TPProblem

            tpp = TPProblem(
                input_irreps,
                sh_irreps,
                tp_irreps,
                instructions=instructions_tp,
                shared_weights=False,
                internal_weights=False,
            )
            self.tp_conv = TensorProductConv(
                tpp, torch_op=True, deterministic=False, use_opaque=False
            )

        else:
            self.tp = o3.TensorProduct(
                input_irreps,
                sh_irreps,
                tp_irreps,
                instructions=instructions_tp,
                shared_weights=False,
                internal_weights=False,
            )

        self.sort = Sort(tp_irreps)

        tp_irreps = self.sort.irreps_sorted

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=torch.nn.SiLU,
            use_bias=False,
            init_scale=mlp_init_scale,
        )

        # add extra irreps to output to account for gate
        gate_irreps = o3.Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
        output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = o3.Linear(irreps_in=tp_irreps.regroup(), irreps_out=output_irreps.regroup())

        # skip connection has per-species weights
        self.skip = o3.Linear(irreps_in=input_irreps.regroup(), irreps_out=output_irreps.regroup())

        if layer_norm:
            self.layer_norm = RMSLayerNorm(
                irreps=output_irreps,
                centering=False,
                std_balance_degrees=True,
            )
        else:
            self.layer_norm = None

        self.gate = Gate(irreps_out=output_irreps)

    def forward(
        self,
        features: torch.Tensor,
        species: torch.Tensor,
        sh: torch.Tensor,
        radial_basis: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        messages = self.linear_1(features)
        radial_message = self.radial_mlp(radial_basis)

        if self.kernel:
            messages_agg = self.sort(
                self.tp_conv(messages, sh, radial_message, receivers, senders)
            ) / torch.sqrt(self.avg_n_neighbors)
        else:
            messages = self.tp(messages[senders], sh, radial_message)
            messages_agg = scatter(
                messages, receivers, dim=0, dim_size=features.size(0)
            ) / torch.sqrt(self.avg_n_neighbors)
            messages_agg = self.sort(messages_agg)

        skip = self.skip(species, features) if self.index_weights else self.skip(features)
        features = self.linear_2(messages_agg) + skip

        if self.layer_norm is not None:
            features = self.layer_norm(features)

        return self.gate(features)


class NequixTorch(torch.nn.Module):
    def __init__(
        self,
        n_species,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "128x0e + 128x1o + 128x2e + 128x3o",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        index_weights: bool = False,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 1.0,
        atom_energies: Optional[Sequence[float]] = None,
        layer_norm: bool = False,
        kernel: bool = False,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.register_buffer("shift", torch.tensor(shift))
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer(
            "atom_energies",
            torch.tensor(atom_energies, dtype=torch.float64)
            if atom_energies is not None
            else torch.zeros(n_species, dtype=torch.float64),
        )

        input_irreps = o3.Irreps(f"{n_species}x0e")
        sh_irreps = o3.Irreps.spherical_harmonics(lmax)
        hidden_irreps = o3.Irreps(hidden_irreps)

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                NequixTorchConvolution(
                    input_irreps=input_irreps if i == 0 else hidden_irreps,
                    output_irreps=hidden_irreps
                    if i < n_layers - 1
                    else hidden_irreps.filter(keep=["0e"]),
                    sh_irreps=sh_irreps,
                    n_species=n_species,
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

        self.readout = o3.Linear(
            irreps_in=hidden_irreps.filter(keep=["0e"]).regroup(), irreps_out="1x0e"
        )

    def node_energies(
        self,
        displacements: torch.Tensor,
        species: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ):
        # input features are one-hot encoded species
        features = torch.nn.functional.one_hot(species, self.n_species).to(torch.float32)
        r_norm = torch.linalg.norm(displacements, ord=2, dim=-1)

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.cutoff)
            * polynomial_cutoff(
                r_norm,
                self.cutoff,
                self.radial_polynomial_p,
            )[:, None]
        )

        # compute spherical harmonics of edge displacements
        sh = o3.spherical_harmonics(
            o3.Irreps.spherical_harmonics(self.lmax),
            displacements,
            normalize=True,
            normalization="component",
        )

        for layer in self.layers:
            features = layer(
                features,
                species,
                sh,
                radial_basis,
                senders,
                receivers,
            )

        node_energies = self.readout(features)

        # scale and shift energies
        node_energies = node_energies * self.scale + self.shift

        # add isolated atom energies to each node as prior
        node_energies = node_energies + self.atom_energies[species, None]

        return node_energies

    def forward(
        self,
        species: torch.Tensor,
        positions: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        cell: torch.Tensor,
        n_node: torch.Tensor,
        n_edge: torch.Tensor,
        n_graph: torch.Tensor,
    ):
        positions.requires_grad_(True)
        if cell is not None:
            eps = torch.zeros_like(cell)
            eps.requires_grad_(True)
            eps_sym = (eps + eps.swapaxes(1, 2)) / 2
            positions_strain = positions + torch.bmm(
                positions.unsqueeze(-2), torch.index_select(eps_sym, 0, n_graph)
            ).squeeze(-2)
            cell_strain = cell + torch.bmm(cell, eps_sym)
            r = torch.index_select(positions_strain, 0, edge_index[0]) - torch.index_select(
                positions_strain, 0, edge_index[1]
            )
            edge_index_batches = torch.index_select(n_graph, 0, edge_index[0])
            # bj <- b1j <- b1j + b1i @ bij
            r = torch.baddbmm(
                r.view(-1, 1, 3),
                edge_attr.view(-1, 1, 3),
                torch.index_select(cell_strain, 0, edge_index_batches),
            ).view(-1, 3)

            # Note: if we try to use senders/receiver from graph_dict then batching gets messed up
            node_energies = self.node_energies(r, species, edge_index[0], edge_index[1])

            minus_forces, virial = torch.autograd.grad(
                outputs=[node_energies.sum()],
                inputs=[positions, eps],
                create_graph=True,
                materialize_grads=True,
            )

            det = torch.abs(torch.linalg.det(cell))[:, None, None]
            stress = virial / det
        else:
            r = positions[edge_index[0]] - positions[edge_index[1]]
            node_energies = self.node_energies(r, species, edge_index[0], edge_index[1])
            minus_forces = torch.autograd.grad(
                outputs=[node_energies.sum()],
                inputs=[positions],
                create_graph=False,
                materialize_grads=False,
            )[0]
            stress = None
        return node_energies[:, 0], -minus_forces, stress


def get_optimizer_param_groups(model, weight_decay):
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters

        # Apply weight decay to weights of Linear layers only
        # Exclude biases and all LayerNorm/BatchNorm parameters
        if "bias" in name:
            no_decay_params.append(param)
        # weight does o3.Linear and weights does MLP
        elif "weight" or "weights" in name:
            # Linear layer weights
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups


def save_model(path: str, model: torch.nn.Module, config: dict):
    """Save a model and its config to a file."""
    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        torch.save(model.state_dict(), f)


def load_model(path: str) -> tuple[NequixTorch, dict]:
    """Load a model and its config from a file."""
    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        model = NequixTorch(
            n_species=len(config["atomic_numbers"]),
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
            atom_energies=[config["atom_energies"][str(n)] for n in config["atomic_numbers"]],
            kernel=config["kernel"],
        )
        state_dict = torch.load(f, map_location="cpu")
        model.load_state_dict(state_dict)
        return model, config
