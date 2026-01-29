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
from nequix.model import Nequix, load_model, save_model, weight_decay_mask


@eqx.filter_jit
def loss(model, batch, energy_weight, force_weight, stress_weight, loss_type="huber"):
    """Return huber loss and MAE of energy and force in eV and eV/Å respectively"""
    energy, forces, stress = model(batch)
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    config = {
        "mse": {"energy": "mse", "force": "mse", "stress": "mse"},
        "huber": {"energy": "huber", "force": "huber", "stress": "huber"},
        "mae": {"energy": "mae", "force": "l2", "stress": "mae"},
    }[loss_type]

    loss_fns = {
        "mae": lambda pred, true: jnp.abs(pred - true),
        "mse": lambda pred, true: (pred - true) ** 2,
        "huber": lambda pred, true: optax.losses.huber_loss(pred, true, delta=0.1),
    }

    # energy per atom (see eq. 30 https://www.nature.com/articles/s41467-023-36329-y)
    # can be achieved by dividing predictied and true energy by number of atoms
    energy_loss_per_atom = jnp.sum(
        loss_fns[config["energy"]](energy / batch.n_node, batch.globals["energy"] / batch.n_node)
        * graph_mask
    ) / jnp.sum(graph_mask)

    if config["force"] == "l2":
        # l2 norm loss for forces
        # NOTE: double where trick is needed to avoid nan's
        force_diff_squared = jnp.sum((forces - batch.nodes["forces"]) ** 2, axis=-1)
        safe_force_diff_squared = jnp.where(force_diff_squared == 0.0, 1.0, force_diff_squared)
        force_loss = jnp.sum(
            jnp.where(force_diff_squared == 0.0, 0.0, jnp.sqrt(safe_force_diff_squared)) * node_mask
        ) / jnp.sum(node_mask)
    else:
        force_loss = jnp.sum(
            loss_fns[config["force"]](forces, batch.nodes["forces"]) * node_mask[:, None]
        ) / (3 * jnp.sum(node_mask))

    if stress_weight > 0:
        stress_loss = jnp.sum(
            loss_fns[config["stress"]](stress, batch.globals["stress"]) * graph_mask[:, None, None]
        ) / (9 * jnp.sum(graph_mask))
    else:
        stress_loss = 0

    total_loss = (
        energy_weight * energy_loss_per_atom
        + force_weight * force_loss
        + stress_weight * stress_loss
    )

    # metrics:

    # MAE energy
    energy_mae_per_atom = jnp.sum(
        jnp.abs(energy / batch.n_node - batch.globals["energy"] / batch.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    # MAE forces
    force_mae = jnp.sum(jnp.abs(forces - batch.nodes["forces"]) * node_mask[:, None]) / (
        3 * jnp.sum(node_mask)
    )

    # MAE stress
    stress_mae_per_atom = jnp.sum(
        jnp.abs(stress - batch.globals["stress"])
        / jnp.where(batch.n_node > 0, batch.n_node, 1.0)[:, None, None]
        * graph_mask[:, None, None]
    ) / (9 * jnp.sum(graph_mask))

    return total_loss, {
        "energy_mae_per_atom": energy_mae_per_atom,
        "force_mae": force_mae,
        "stress_mae_per_atom": stress_mae_per_atom,
    }


def evaluate(
    model, dataloader, energy_weight=1.0, force_weight=1.0, stress_weight=1.0, loss_type="huber"
):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    total_metrics = defaultdict(int)
    total_count = 0
    for batch in prefetch(dataloader):
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, metrics = loss(
            model, batch, energy_weight, force_weight, stress_weight, loss_type
        )
        total_metrics["loss"] += val_loss * n_graphs
        for key, value in metrics.items():
            total_metrics[key] += value * n_graphs
        total_count += n_graphs

    for key, value in total_metrics.items():
        total_metrics[key] = value / total_count

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
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
    )
    if "finetune_from" in config and Path(config["finetune_from"]).exists():
        if "atom_energies" in config:
            # TODO
            raise NotImplementedError("Updating atom energies not implemented for JAX backend")
        model, _ = load_model(config["finetune_from"])

    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(model, eqx.is_array))[0])

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

    # @eqx.filter_jit
    @functools.partial(eqx.filter_pmap, in_axes=(0, 0, None, 0, 0), axis_name="device")
    def train_step(model, ema_model, step, opt_state, batch):
        # training step
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model,
            batch,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
        )
        grads = jax.lax.pmean(grads, axis_name="device")
        metrics["grad_norm"] = optax.global_norm(grads)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)

        # update EMA model
        # don't weight early steps as much (from https://github.com/fadel/pytorch_ema)
        decay = jnp.minimum(config["ema_decay"], (1 + step) / (10 + step))
        ema_params, ema_static = eqx.partition(ema_model, eqx.is_array)
        model_params = eqx.filter(model, eqx.is_array)
        new_ema_params = jax.tree.map(
            lambda ep, mp: ep * decay + mp * (1 - decay), ema_params, model_params
        )
        ema_model = eqx.combine(ema_static, new_ema_params)

        return (
            model,
            ema_model,
            opt_state,
            total_loss,
            metrics,
        )

    for epoch in range(start_epoch, config["n_epochs"]):
        start_time = time.time()
        train_loader.loader.set_epoch(epoch)
        for batch in prefetch(train_loader):
            batch_time = time.time() - start_time
            start_time = time.time()
            (model, ema_model, opt_state, total_loss, metrics) = train_step(
                model, ema_model, step, opt_state, batch
            )
            train_time = time.time() - start_time
            step = step + 1
            if step % config["log_every"] == 0:
                logs = {}
                logs["train/loss"] = total_loss.mean().item()
                logs["learning_rate"] = schedule(step).item()
                logs["train/batch_time"] = batch_time
                logs["train/train_time"] = train_time
                for key, value in metrics.items():
                    logs[f"train/{key}"] = value.mean().item()
                logs["train/batch_size"] = (
                    jax.vmap(jraph.get_graph_padding_mask)(batch).sum().item()
                )
                wandb.log(logs, step=step)
                print(f"step: {step}, logs: {logs}")
                wandb_sync()
            start_time = time.time()

        ema_model_single = jax.tree.map(lambda x: x[0], ema_model)
        val_metrics = evaluate(
            ema_model_single,
            val_loader,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
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
        for key, value in val_metrics.items():
            logs[f"val/{key}"] = value.item()
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
