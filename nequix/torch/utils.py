from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch
import jax
import jax.numpy as jnp
import equinox as eqx
from nequix.torch.model import NequixTorch
from nequix.model import Nequix


# copied from https://github.com/facebookresearch/vissl/blob/09270ed25a6c2cf71263d955b64cbe076d34ac45/vissl/data/data_helper.py#L93
class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size: int, seed: int = 0, **kwargs):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.
        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, seed=seed, **kwargs)  # type: ignore

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(range((self.rank * self.num_samples), (self.rank + 1) * self.num_samples))
        )[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter


def convert_layer_torch_to_jax(layer_idx, torch_model, jax_model):
    # Linear 1
    for (slice_1D, shape_2D), (weight_name, _) in zip(
        torch_model.layers[layer_idx].linear_1.weight_index_slices,
        jax_model.layers[layer_idx].linear_1._weights.items(),
    ):
        jax_model.layers[layer_idx].linear_1._weights[weight_name] = jnp.asarray(
            torch_model.layers[layer_idx]
            .linear_1.weight[slice_1D]
            .reshape(shape_2D)
            .detach()
            .cpu()
            .numpy()
        )

    # Radial MLP
    for i, (torch_layer, jax_layer) in enumerate(
        zip(
            torch_model.layers[layer_idx].radial_mlp.layers,
            jax_model.layers[layer_idx].radial_mlp.layers,
        )
    ):
        jax_model.layers[layer_idx].radial_mlp.layers[i] = eqx.tree_at(
            lambda x: x.weights, jax_layer, np.array(torch_layer.weights.detach().cpu().numpy())
        )

    # Skip
    for (slice_1D, shape_2D), (weight_name, _) in zip(
        torch_model.layers[layer_idx].skip.weight_index_slices,
        jax_model.layers[layer_idx].skip._weights.items(),
    ):
        jax_model.layers[layer_idx].skip._weights[weight_name] = jnp.asarray(
            torch_model.layers[layer_idx]
            .skip.weight[slice_1D]
            .reshape(shape_2D)
            .detach()
            .cpu()
            .numpy()
        )

    # Linear 2
    for (slice_1D, shape_2D), (weight_name, _) in zip(
        torch_model.layers[layer_idx].linear_2.weight_index_slices,
        jax_model.layers[layer_idx].linear_2._weights.items(),
    ):
        jax_model.layers[layer_idx].linear_2._weights[weight_name] = jnp.asarray(
            torch_model.layers[layer_idx]
            .linear_2.weight[slice_1D]
            .reshape(shape_2D)
            .detach()
            .cpu()
            .numpy()
        )

    # Layer norm
    for i, (affine_weight_torch, affine_weight_jax) in enumerate(
        zip(
            torch_model.layers[layer_idx].layer_norm.affine_weight,
            jax_model.layers[layer_idx].layer_norm.affine_weight,
        )
    ):
        jax_model.layers[layer_idx].layer_norm.affine_weight[i] = eqx.tree_at(
            lambda x: x, affine_weight_jax, np.array(affine_weight_torch.detach().cpu().numpy())
        )

    return jax_model


def convert_model_torch_to_jax(torch_model, config):
    jax_model = Nequix(
        key=jax.random.key(0),
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
    )
    for layer_idx in range(len(torch_model.layers)):
        jax_model = convert_layer_torch_to_jax(layer_idx, torch_model, jax_model)

    for (weight_name, _), (slice_1D, shape_2D) in zip(
        jax_model.readout._weights.items(), torch_model.readout.weight_index_slices
    ):
        jax_model.readout._weights[weight_name] = jnp.asarray(
            torch_model.readout.weight[slice_1D].reshape(shape_2D).detach().cpu().numpy()
        )

    return jax_model, config


def convert_layer_jax_to_torch(layer_idx, jax_model, torch_model):
    # Linear 1
    jax_linear_1_weights = jnp.concatenate(
        [x.flatten() for x in jax_model.layers[layer_idx].linear_1._weights.values()]
    )
    torch_model.layers[layer_idx].linear_1.weight = torch.nn.Parameter(
        torch.from_numpy(np.array(jax_linear_1_weights))
    )

    # Radial MLP
    for torch_layer, jax_layer in zip(
        torch_model.layers[layer_idx].radial_mlp.layers,
        jax_model.layers[layer_idx].radial_mlp.layers,
    ):
        torch_layer.weights = torch.nn.Parameter(torch.from_numpy(np.array(jax_layer.weights)))

    # Skip
    jax_linear_skip_weights = jnp.concatenate(
        [x.flatten() for x in jax_model.layers[layer_idx].skip._weights.values()]
    )
    torch_model.layers[layer_idx].skip.weight = torch.nn.Parameter(
        torch.from_numpy(np.array(jax_linear_skip_weights))
    )

    # Linear 2
    jax_linear_2_weights = jnp.concatenate(
        [x.flatten() for x in jax_model.layers[layer_idx].linear_2._weights.values()]
    )
    torch_model.layers[layer_idx].linear_2.weight = torch.nn.Parameter(
        torch.from_numpy(np.array(jax_linear_2_weights))
    )

    # Layer norm
    for i in range(len(torch_model.layers[layer_idx].layer_norm.affine_weight)):
        torch_model.layers[layer_idx].layer_norm.affine_weight[i] = torch.nn.Parameter(
            torch.from_numpy(np.array(jax_model.layers[layer_idx].layer_norm.affine_weight[i]))
        )

    return torch_model


def convert_model_jax_to_torch(jax_model, config, use_kernel):
    torch_model = NequixTorch(
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
        kernel=use_kernel,
    )
    for layer_idx in range(len(jax_model.layers)):
        torch_model = convert_layer_jax_to_torch(layer_idx, jax_model, torch_model)

    # Readout
    jax_linear_readout_weights = jnp.concatenate(
        [x.flatten() for x in jax_model.readout._weights.values()]
    )
    torch_model.readout.weight = torch.nn.Parameter(
        torch.from_numpy(np.array(jax_linear_readout_weights))
    )

    return torch_model, config
