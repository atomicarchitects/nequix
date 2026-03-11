import argparse
import functools
import os
import time
from collections import defaultdict
from pathlib import Path

import cloudpickle
import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import optax
import yaml
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from nequix.data import (
    AseDBDataset,
    ConcatDataset,
    DataLoader,
    ParallelLoader,
    average_atom_energies,
    dataset_stats,
    prefetch,
)
from nequix.model import Nequix, load_model, save_model, weight_decay_mask, node_graph_idx


def sample_sigma(key, n_graphs, sigma_min, sigma_max, p_clean=0.1):
    """
    Sample noise levels. With probability p_clean, sigma=0 (supervised).
    Otherwise log-uniform between sigma_min and sigma_max.
    """
    key_choice, key_sigma = jax.random.split(key)
    log_sigma = jax.random.uniform(
        key_sigma, shape=(n_graphs,),
        minval=jnp.log(sigma_min),
        maxval=jnp.log(sigma_max),
    )
    sigma = jnp.exp(log_sigma)
    # randomly zero out some graphs -> supervised mode for those
    is_clean = jax.random.uniform(key_choice, shape=(n_graphs,)) < p_clean
    return jnp.where(is_clean, 0.0, sigma)


def add_noise_to_batch(
    batch: jraph.GraphsTuple,
    sigma: jax.Array,          # (n_graphs,)
    key: jax.Array,
    node_graph_index: jax.Array,
) -> tuple[jraph.GraphsTuple, jax.Array]:
    """Perturb positions; graphs with sigma=0 are left untouched."""
    sigma_per_node = sigma[node_graph_index]             # (n_nodes,)
    noise = jax.random.normal(key, batch.nodes["positions"].shape)
    noisy_positions = (
        batch.nodes["positions"] + sigma_per_node[:, None] * noise
    )
    noisy_nodes = {**batch.nodes, "positions": noisy_positions}
    return batch._replace(nodes=noisy_nodes), noise        # noise needed for score target


@eqx.filter_jit
def loss(
    model,
    batch,
    noise,                   # (n_nodes, 3) — the actual Gaussian noise added
    sigma,                   # (n_graphs,)  — 0.0 means supervised
    energy_weight,
    force_weight,
    stress_weight,
    loss_type="huber",
):
    energy, forces, stress = model(batch, sigma=sigma)

    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask  = jraph.get_node_padding_mask(batch)
    node_graph_index = node_graph_idx(batch)               # (n_nodes,)

    sigma_per_node  = sigma[node_graph_index]              # (n_nodes,)
    sigma_per_graph = sigma                               # (n_graphs,)

    is_clean_node  = sigma_per_node  == 0.0              # (n_nodes,)  bool
    is_clean_graph = sigma_per_graph == 0.0              # (n_graphs,) bool

    huber = lambda p, t: optax.losses.huber_loss(p, t, delta=0.1)
    mae   = lambda p, t: jnp.abs(p - t)
    loss_fn = {"huber": huber, "mae": mae, "mse": lambda p, t: (p - t) ** 2}[loss_type]

    # ------------------------------------------------------------------
    # Energy loss — supervised graphs only, per-atom normalised
    # ------------------------------------------------------------------
    energy_residual = loss_fn(
        energy / batch.n_node,
        batch.globals["energy"] / batch.n_node,
    )                                                      # (n_graphs,)
    energy_loss = jnp.sum(
        energy_residual * graph_mask * is_clean_graph
    ) / jnp.maximum(jnp.sum(graph_mask * is_clean_graph), 1.0)

    # ------------------------------------------------------------------
    # Force loss — two terms, masked separately
    # ------------------------------------------------------------------

    # Term 1: supervised (σ=0) — DFT force targets
    dft_force_residual = loss_fn(forces, batch.nodes["forces"])  # (n_nodes, 3)
    supervised_force_loss = jnp.sum(
        dft_force_residual * (node_mask * is_clean_node)[:, None]
    ) / jnp.maximum(3 * jnp.sum(node_mask * is_clean_node), 1.0)

    # Term 2: score matching (σ>0) — target is -noise/sigma (the score)
    # F = -∇U(x,σ) should approximate ∇log p(x_noisy|x_clean) = -noise/σ
    # Weight by σ² so loss is scale-invariant (Karras et al. 2022 eq. 4)
    safe_sigma = jnp.where(sigma_per_node > 0.0, sigma_per_node, 1.0)
    score_targets = -noise / safe_sigma[:, None]                 # (n_nodes, 3)
    score_weight  = sigma_per_node ** 2                          # (n_nodes,)  λ(σ)
    noisy_mask    = node_mask * ~is_clean_node
    score_residual = loss_fn(forces, score_targets)              # (n_nodes, 3)
    score_force_loss = jnp.sum(
        score_residual * score_weight[:, None] * noisy_mask[:, None]
    ) / jnp.maximum(3 * jnp.sum(noisy_mask), 1.0)

    # ------------------------------------------------------------------
    # Stress loss — supervised graphs only
    # ------------------------------------------------------------------
    if stress_weight > 0:
        stress_loss = jnp.sum(
            loss_fn(stress, batch.globals["stress"])
            * (graph_mask * is_clean_graph)[:, None, None]
        ) / jnp.maximum(9 * jnp.sum(graph_mask * is_clean_graph), 1.0)
    else:
        stress_loss = jnp.array(0.0)

    total_loss = (
        energy_weight * energy_loss
        + force_weight * (supervised_force_loss + score_force_loss)
        + stress_weight * stress_loss
    )

    # ------------------------------------------------------------------
    # Metrics (always MAE, for monitoring)
    # ------------------------------------------------------------------
    energy_mae = jnp.sum(
        jnp.abs(energy / batch.n_node - batch.globals["energy"] / batch.n_node)
        * graph_mask * is_clean_graph
    ) / jnp.maximum(jnp.sum(graph_mask * is_clean_graph), 1.0)

    force_mae = jnp.sum(
        jnp.abs(forces - batch.nodes["forces"]) * (node_mask * is_clean_node)[:, None]
    ) / jnp.maximum(3 * jnp.sum(node_mask * is_clean_node), 1.0)

    score_mae = jnp.sum(
        jnp.abs(forces - score_targets) * noisy_mask[:, None]
    ) / jnp.maximum(3 * jnp.sum(noisy_mask), 1.0)

    return total_loss, {
        "energy_mae_per_atom":  energy_mae,
        "force_mae":            force_mae,         # supervised force MAE
        "score_mae":            score_mae,          # score matching MAE
        "stress_mae_per_atom":  stress_loss,
        "frac_clean":           is_clean_graph.mean(),
    }




def evaluate(
    model, dataloader, key, energy_weight=1.0, force_weight=1.0, stress_weight=1.0, loss_type="huber",
    sigma_min: float = 0.001, sigma_max: float = 1.0, p_clean: float = 0.1,
):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    total_metrics = defaultdict(int)
    total_count = 0
    for batch in prefetch(dataloader):
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        node_graph_index = node_graph_idx(batch)

        key, sigma_key, noise_key = jax.random.split(key, 3)
        sigma = sample_sigma(sigma_key, batch.n_node.shape[0], sigma_min, sigma_max, p_clean)
        noisy_batch, noise = add_noise_to_batch(batch, sigma, noise_key, node_graph_index)

        val_loss, metrics = loss(
            model, noisy_batch, noise, sigma, energy_weight, force_weight, stress_weight, loss_type
        )
        total_metrics["loss"] += val_loss * n_graphs
        for metric_key, value in metrics.items():
            total_metrics[metric_key] += value * n_graphs
        total_count += n_graphs

    for metric_key, value in total_metrics.items():
        total_metrics[metric_key] = value / total_count

    return total_metrics


def save_training_state(
    path, model, ema_model, optim, opt_state, step, epoch, best_val_loss, wandb_run_id=None
):
    state = {
        "model": model,
        "ema_model": ema_model,
        "optim": optim,
        "opt_state": opt_state,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
    }
    with open(path, "wb") as f:
        cloudpickle.dump(state, f)


def load_training_state(path):
    with open(path, "rb") as f:
        state = cloudpickle.load(f)
    return (
        state["model"],
        state["ema_model"],
        state["optim"],
        state["opt_state"],
        state["step"],
        state["epoch"],
        state["best_val_loss"],
        state.get("wandb_run_id"),
    )


def train(config_path: str):
    """Train a Nequix model from a config file. See configs/nequix-mp-1.yaml for an example."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # use TMPDIR for slurm jobs if available
    config["cache_dir"] = config.get("cache_dir") or os.environ.get("TMPDIR")

    if isinstance(config["train_path"], list):
        train_dataset = ConcatDataset(
            [
                AseDBDataset(
                    file_path=path,
                    atomic_numbers=config["atomic_numbers"],
                    cutoff=config["cutoff"],
                    backend="jax",
                )
                for path in config["train_path"]
            ]
        )
    else:
        train_dataset = AseDBDataset(
            file_path=config["train_path"],
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
            backend="jax",
        )
    if "valid_frac" in config:
        train_dataset, val_dataset = train_dataset.split(valid_frac=config["valid_frac"])
    else:
        assert "valid_path" in config, "valid_path must be specified if valid_frac is not provided"
        val_dataset = AseDBDataset(
            file_path=config["valid_path"],
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
            backend="jax",
        )

    if "atom_energies" in config:
        atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    else:
        atom_energies = average_atom_energies(train_dataset)

    stats_keys = [
        "shift",
        "scale",
        "avg_n_neighbors",
        "max_n_edges",
        "max_n_nodes",
        "avg_n_nodes",
        "avg_n_edges",
    ]
    if all(key in config for key in stats_keys):
        stats = {key: config[key] for key in stats_keys}
    else:
        stats = dataset_stats(train_dataset, atom_energies)

    num_devices = len(jax.devices())
    print(f"Using {num_devices} devices for training")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=16,
    )
    train_loader = ParallelLoader(train_loader, num_devices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=16,
    )

    wandb_sync = (
        TriggerWandbSyncHook() if os.environ.get("WANDB_MODE") == "offline" else lambda: None
    )

    key = jax.random.key(0)
    model = Nequix(
        key,
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
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
        kernel=config["kernel"],
        add_repulsion=config.get("add_repulsion", False),
    )
    mask = weight_decay_mask(model)
    print(jax.tree.map(lambda x: x, mask))

    if "finetune_from" in config and Path(config["finetune_from"]).exists():
        if "atom_energies" in config:
            # TODO
            raise NotImplementedError("Updating atom energies not implemented for JAX backend")
        model, _ = load_model(config["finetune_from"])

    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])
    print(f"Loaded model with {param_count} parameters")

    # NB: this is not exact because of dynamic batching but should be close enough
    steps_per_epoch = len(train_dataset) // (config["batch_size"] * jax.device_count())
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=config["learning_rate"] * config["warmup_factor"],
        peak_value=config["learning_rate"],
        end_value=1e-6,
        warmup_steps=config["warmup_epochs"] * steps_per_epoch,
        decay_steps=config["n_epochs"] * steps_per_epoch,
    )

    if config["optimizer"] == "adamw":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=config["weight_decay"],
                mask=weight_decay_mask(model),
            ),
        )
    elif config["optimizer"] == "muon":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.contrib.muon(
                learning_rate=schedule,
                weight_decay=config["weight_decay"] if config["weight_decay"] != 0.0 else None,
                weight_decay_mask=weight_decay_mask(model),
            ),
        )
    else:
        raise ValueError(f"optimizer {config['optimizer']} not supported")

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    keys = jax.random.split(key, num_devices)
    keys = jax.device_put_sharded(list(keys), jax.devices())
    model = jax.device_put_replicated(model, list(jax.devices()))
    opt_state = jax.device_put_replicated(opt_state, list(jax.devices()))
    ema_model = jax.tree.map(lambda x: x.copy(), model)  # copy model
    step = jnp.array(0)
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None

    if "resume_from" in config and Path(config["resume_from"]).exists():
        (
            model,
            ema_model,
            optim,
            opt_state,
            step,
            start_epoch,
            best_val_loss,
            wandb_run_id,
        ) = load_training_state(config["resume_from"])

    wandb_init_kwargs = {"project": "nequix", "config": config}
    if wandb_run_id:
        wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})
    wandb.init(**wandb_init_kwargs)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.run.summary["param_count"] = param_count
        wandb_run_id = getattr(wandb.run, "id", None)

    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0, 0), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch, key):
        # sample sigma and noise inside pmap so each device gets its own key
        key, sigma_key, noise_key = jax.random.split(key, 3)
        n_graphs = batch.n_node.shape[0]
        node_graph_index = node_graph_idx(batch)

        sigma = sample_sigma(
            sigma_key, n_graphs,
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"],
            p_clean=config["p_clean"],
        )
        noisy_batch, noise = add_noise_to_batch(batch, sigma, noise_key, node_graph_index)

        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, noisy_batch, noise, sigma,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        metrics["grad_norm"] = optax.global_norm(grads)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        decay = jnp.minimum(config["ema_decay"], (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)
        return model, ema_model, opt_state, total_loss, metrics


    for epoch in range(start_epoch, config["n_epochs"]):
        start_time = time.time()
        train_loader.loader.set_epoch(epoch)
        for batch in prefetch(train_loader):
            batch_time = time.time() - start_time
            start_time = time.time()
            
            keys, step_keys = jax.vmap(jax.random.split)(keys).swapaxes(0, 1)
            (model, ema_model, opt_state, total_loss, metrics) = train_step(
                model, ema_model, step, opt_state, batch, step_keys
            )

            # jax.block_until_ready(model)
            train_time = time.time() - start_time
            step = step + 1
            if step % config["log_every"] == 0:
                logs = {}
                logs["train/loss"] = total_loss.mean().item()
                logs["learning_rate"] = schedule(step).item()
                logs["train/batch_time"] = batch_time
                logs["train/train_time"] = train_time
                for metric_key, value in metrics.items():
                    logs[f"train/{metric_key}"] = value.mean().item()
                logs["train/batch_size"] = (
                    jax.vmap(jraph.get_graph_padding_mask)(batch).sum().item()
                )
                wandb.log(logs, step=step)
                print(f"step: {step}, logs: {logs}")
                wandb_sync()
            start_time = time.time()

        ema_model_single = jax.tree.map(lambda x: x[0], ema_model)
        eval_key = jax.random.fold_in(keys[0], epoch)
        val_metrics = evaluate(
            ema_model_single,
            val_loader,
            eval_key,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
            sigma_min=config["sigma_min"],
            sigma_max=config["sigma_max"],
            p_clean=config["p_clean"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_model(Path(wandb.run.dir) / "checkpoint.nqx", ema_model_single, config)

        save_training_state(
            Path(wandb.run.dir) / "state.pkl",
            model,
            ema_model,
            optim,
            opt_state,
            step,
            epoch + 1,
            best_val_loss,
            wandb_run_id=wandb_run_id,
        )

        if "state_path" in config:
            save_training_state(
                config["state_path"],
                model,
                ema_model,
                optim,
                opt_state,
                step,
                epoch + 1,
                best_val_loss,
                wandb_run_id=wandb_run_id,
            )

        logs = {}
        for metric_key, value in val_metrics.items():
            logs[f"val/{metric_key}"] = value.item()
        logs["epoch"] = epoch
        wandb.log(logs, step=step)
        print(f"epoch: {epoch}, logs: {logs}")
        wandb_sync()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    main()
