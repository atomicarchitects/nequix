import argparse
import itertools
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
import optax
import yaml
from nequix.data import (
    AseDBDataset,
    ConcatDataset,
    DataLoader,
    dataset_stats,
    prefetch,
)
from nequix.model import load_model, node_graph_idx, save_model, weight_decay_mask
from nequix.train import evaluate as efs_evaluate
from nequix.pft.data import PhononDataset
from nequix.train import loss as efs_loss, save_training_state, load_training_state

import wandb


def loss(
    model,
    graph,
    energy_weight=20.0,
    force_weight=20.0,
    stress_weight=5.0,
    hessian_weight=100.0,
    checkpoint_grad_energy=False,
):
    def total_energy_fn(positions_eps: tuple[jax.Array, jax.Array]):
        positions, eps = positions_eps
        eps_sym = (eps + eps.swapaxes(1, 2)) / 2
        eps_sym_per_node = jnp.repeat(
            eps_sym,
            graph.n_node,
            axis=0,
            total_repeat_length=graph.nodes["positions"].shape[0],
        )
        positions = positions + jnp.einsum("ik,ikj->ij", positions, eps_sym_per_node)
        cell_per_edge = jnp.repeat(
            graph.globals["cell"],
            graph.n_edge,
            axis=0,
            total_repeat_length=graph.edges["shifts"].shape[0],
        )
        offsets = jnp.einsum("ij,ijk->ik", graph.edges["shifts"], cell_per_edge)
        r = positions[graph.senders] - positions[graph.receivers] + offsets
        node_energies = model.node_energies(
            r, graph.nodes["species"], graph.senders, graph.receivers
        )
        return jnp.sum(node_energies), node_energies

    eps = jnp.zeros_like(graph.globals["cell"])

    node_mask = jraph.get_node_padding_mask(graph)
    graph_mask = jraph.get_graph_padding_mask(graph)

    if checkpoint_grad_energy:
        # checkpoint grad energy (useful for larger models/smaller GPUs)
        grad_energy = jax.grad(jax.checkpoint(total_energy_fn), has_aux=True)
    else:
        grad_energy = jax.grad(total_energy_fn, has_aux=True)
    (minus_forces, virial), node_energies = grad_energy((graph.nodes["positions"], eps))

    # hessian column is jacobian vector product of grad energy w.r.t. node
    # degree of freedoms in vs
    # _, hvp, _ = jax.linearize(grad_energy, (graph.nodes["positions"], eps), has_aux=True)
    # hvp = jax.jit(hvp)
    # hessian_col, _ = hvp((graph.nodes["vs"], eps))
    hessian_col, _ = jax.jvp(
        grad_energy,
        ((graph.nodes["positions"], eps),),
        ((graph.nodes["vs"], eps),),
        has_aux=True,
    )[1]

    graph_energies = jraph.segment_sum(
        node_energies,
        node_graph_idx(graph),
        num_segments=graph.n_node.shape[0],
        indices_are_sorted=True,
    )
    det = jnp.abs(jnp.linalg.det(graph.globals["cell"]))[:, None, None]
    det = jnp.where(det > 0.0, det, 1.0)  # padded graphs have det = 0
    stress = virial / det

    # mask out padding nodes
    hessian_col = jnp.where(node_mask[:, None], hessian_col, 0.0)
    minus_forces = jnp.where(node_mask[:, None], minus_forces, 0.0)
    stress = jnp.where(graph_mask[:, None, None], stress, 0.0)

    energy = graph_energies[:, 0]
    forces = -minus_forces

    energy_loss_per_atom = jnp.sum(
        jnp.abs(energy / graph.n_node - graph.globals["energy"] / graph.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    force_diff_squared = jnp.sum((forces - graph.nodes["forces"]) ** 2, axis=-1)
    safe_force_diff_squared = jnp.where(force_diff_squared == 0.0, 1.0, force_diff_squared)
    force_loss = jnp.sum(
        jnp.where(force_diff_squared == 0.0, 0.0, jnp.sqrt(safe_force_diff_squared)) * node_mask
    ) / jnp.sum(node_mask)

    stress_loss = jnp.sum(jnp.abs(stress - graph.globals["stress"]) * graph_mask[:, None, None]) / (
        9 * jnp.sum(graph_mask)
    )

    # MAE
    hessian_loss = jnp.sum(
        jnp.abs(hessian_col - graph.nodes["hessian_col"]) * node_mask[:, None]
    ) / (3 * jnp.sum(node_mask))

    total_loss = (
        energy_weight * energy_loss_per_atom
        + force_weight * force_loss
        + stress_weight * stress_loss
        + hessian_weight * hessian_loss
    )

    # metrics:
    energy_mae_per_atom = jnp.sum(
        jnp.abs(energy / graph.n_node - graph.globals["energy"] / graph.n_node) * graph_mask
    ) / jnp.sum(graph_mask)

    # MAE forces
    force_mae = jnp.sum(jnp.abs(forces - graph.nodes["forces"]) * node_mask[:, None]) / (
        3 * jnp.sum(node_mask)
    )

    # MAE stress
    stress_mae_per_atom = jnp.sum(
        jnp.abs(stress - graph.globals["stress"])
        / jnp.where(graph.n_node > 0, graph.n_node, 1.0)[:, None, None]
        * graph_mask[:, None, None]
    ) / (9 * jnp.sum(graph_mask))

    # MAE hessian
    hessian_mae_per_atom = jnp.sum(
        jnp.abs(hessian_col - graph.nodes["hessian_col"]) * node_mask[:, None]
    ) / (3 * jnp.sum(node_mask))

    return total_loss, {
        "energy_mae_per_atom": energy_mae_per_atom,
        "force_mae": force_mae,
        "stress_mae_per_atom": stress_mae_per_atom,
        "hessian_mae_per_atom": hessian_mae_per_atom,
    }


def evaluate(
    model,
    dataloader,
    energy_weight=20.0,
    force_weight=20.0,
    stress_weight=5.0,
    hessian_weight=100.0,
    checkpoint_grad_energy=False,
):
    total_metrics = defaultdict(int)
    total_count = 0
    for batch in prefetch(dataloader):
        n_graphs = jnp.sum(jraph.get_graph_padding_mask(batch))
        val_loss, metrics = eqx.filter_jit(loss)(
            model,
            batch,
            energy_weight,
            force_weight,
            stress_weight,
            hessian_weight,
            checkpoint_grad_energy,
        )
        total_metrics["loss"] += val_loss * n_graphs
        for k, v in metrics.items():
            total_metrics[k] += v * n_graphs
        total_count += n_graphs

    for k, v in total_metrics.items():
        total_metrics[k] = v / total_count
    return total_metrics


def train(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model, original_config = load_model(config["finetune_from"])

    if config["optimizer"] == "muon":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.contrib.muon(
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"] if config["weight_decay"] != 0.0 else None,
                weight_decay_mask=weight_decay_mask(model),
            ),
        )
    elif config["optimizer"] == "adam":
        optim = optax.chain(
            optax.clip_by_global_norm(config["grad_clip_norm"]),
            optax.adamw(
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"] if config["weight_decay"] != 0.0 else None,
                mask=weight_decay_mask(model),
            ),
        )

    ema_model = jax.tree.map(lambda x: x.copy(), model)  # copy model

    opt_state = optim.init(model)

    if "displacement" in config and config["displacement"]:
        train_dataset = AseDBDataset(
            file_path=config["train_path"],
            atomic_numbers=original_config["atomic_numbers"],
            cutoff=original_config["cutoff"],
        )
    else:
        train_dataset = PhononDataset(
            file_path=config["train_path"],
            atomic_numbers=original_config["atomic_numbers"],
            cutoff=original_config["cutoff"],
            random_col=True,
        )

    val_dataset = PhononDataset(
        file_path=config["val_path"],
        atomic_numbers=original_config["atomic_numbers"],
        cutoff=original_config["cutoff"],
        random_col=False,  # always use first column for validation
    )

    if isinstance(config["extra_train_path"], list):
        extra_train_dataset = ConcatDataset(
            [
                AseDBDataset(
                    file_path=path,
                    atomic_numbers=original_config["atomic_numbers"],
                    cutoff=original_config["cutoff"],
                )
                for path in config["extra_train_path"]
            ]
        )
    else:
        extra_train_dataset = AseDBDataset(
            file_path=config["extra_train_path"],
            atomic_numbers=original_config["atomic_numbers"],
            cutoff=original_config["cutoff"],
        )
    if "extra_val_frac" in config:
        extra_train_dataset, extra_val_dataset = extra_train_dataset.split(
            valid_frac=config["extra_val_frac"]
        )
    else:
        extra_val_dataset = AseDBDataset(
            file_path=config["extra_val_path"],
            cutoff=original_config["cutoff"],
            atomic_numbers=original_config["atomic_numbers"],
        )

    stats_keys = [
        # "shift",  shift, scale, avg_n_neighbors are used from the pretraining model
        # "scale",
        # "avg_n_neighbors",
        "max_n_edges",
        "max_n_nodes",
        "avg_n_nodes",
        "avg_n_edges",
    ]
    if all(key in config for key in stats_keys):
        stats = {key: config[key] for key in stats_keys}
    else:
        atom_energies = [
            original_config["atom_energies"][str(n)] for n in original_config["atomic_numbers"]
        ]
        stats = dataset_stats(train_dataset, atom_energies)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        n_graph=config.get("n_graph", None),
        shuffle=True,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        n_graph=config.get("n_graph", None),
        shuffle=False,
        max_n_nodes=stats["max_n_nodes"],
        max_n_edges=stats["max_n_edges"],
        avg_n_nodes=stats["avg_n_nodes"],
        avg_n_edges=stats["avg_n_edges"],
        num_workers=8,
    )

    extra_val_loader = DataLoader(
        extra_val_dataset,
        batch_size=original_config["batch_size"],
        shuffle=False,
        max_n_nodes=original_config["max_n_nodes"],
        max_n_edges=original_config["max_n_edges"],
        avg_n_nodes=original_config["avg_n_nodes"],
        avg_n_edges=original_config["avg_n_edges"],
        num_workers=8,
    )

    extra_train_loader = DataLoader(
        extra_train_dataset,
        batch_size=original_config["batch_size"],
        shuffle=True,
        max_n_nodes=original_config["max_n_nodes"],
        max_n_edges=original_config["max_n_edges"],
        avg_n_nodes=original_config["avg_n_nodes"],
        avg_n_edges=original_config["avg_n_edges"],
        num_workers=8,
    )

    if "displacement" in config and config["displacement"]:
        loss_fn = partial(
            efs_loss,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            stress_weight=config["stress_weight"],
        )
    else:
        loss_fn = partial(
            loss,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            stress_weight=config["stress_weight"],
            hessian_weight=config["hessian_weight"],
            checkpoint_grad_energy=config["checkpoint_grad_energy"],
        )

    extra_train_loss_fn = partial(
        efs_loss,
        energy_weight=config.get("extra_energy_weight", original_config["energy_weight"]),
        force_weight=config.get("extra_force_weight", original_config["force_weight"]),
        stress_weight=config.get("extra_stress_weight", original_config["stress_weight"]),
    )

    @eqx.filter_jit(donate="all")
    def train_step(model, ema_model, loss_fn, step, opt_state, graph):
        (total_loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, graph
        )
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
        return model, ema_model, opt_state, total_loss, metrics

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

    wandb_init_kwargs = {"project": "nequix-phonon", "config": config}
    if wandb_run_id:
        wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})
    wandb.init(**wandb_init_kwargs)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb_run_id = getattr(wandb.run, "id", None)

    for epoch in range(start_epoch, config["n_epochs"]):
        train_loader.set_epoch(epoch)
        extra_train_loader.set_epoch(epoch)

        # infinite iterator over extra train loader
        extra_iter = itertools.chain.from_iterable(iter(lambda: prefetch(extra_train_loader), None))

        for batch in prefetch(train_loader):
            start = time.time()
            model, ema_model, opt_state, total_loss, metrics = train_step(
                model, ema_model, loss_fn, step.copy(), opt_state, batch
            )
            total_loss.block_until_ready()
            step_time = time.time() - start

            # extra train step with original training data/loss fn
            for _ in range(config["extra_train_steps"]):
                extra_batch = next(extra_iter)
                model, ema_model, opt_state, extra_total_loss, extra_metrics = train_step(
                    model, ema_model, extra_train_loss_fn, step.copy(), opt_state, extra_batch
                )

            step = step + 1
            if step % config["log_every"] == 0:
                logs = {}
                for k, v in metrics.items():
                    logs[f"train/{k}"] = v.mean().item()
                logs["train/loss"] = total_loss.mean().item()
                logs["train/batch_size"] = jraph.get_graph_padding_mask(batch).sum().item()
                logs["train/time"] = step_time
                if config["extra_train_steps"]:
                    for k, v in extra_metrics.items():
                        logs[f"extra_train/{k}"] = v.mean().item()
                    logs["extra_train/loss"] = extra_total_loss.mean().item()
                wandb.log(logs, step=step)
                print(f"step {step:03d} logs: {logs}")

        if epoch % config["val_every"] == 0 or epoch == config["n_epochs"] - 1:
            val_metrics = evaluate(
                ema_model,
                val_loader,
                config["energy_weight"],
                config["force_weight"],
                config["stress_weight"],
                config["hessian_weight"],
                config["checkpoint_grad_energy"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_model(
                    Path(wandb.run.dir) / "checkpoint.nqx", ema_model, original_config | config
                )

            # always save training state to wandb dir
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
            # also save training state to config state_path
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
            for k, v in val_metrics.items():
                logs[f"val/{k}"] = v.mean().item()
            logs["val/loss"] = val_metrics["loss"].mean().item()
            logs["epoch"] = epoch
            wandb.log(logs, step=step)
            print(f"epoch {epoch:03d} val metrics: {logs}")

            extra_val_metrics = efs_evaluate(
                ema_model,
                extra_val_loader,
                energy_weight=original_config["energy_weight"],
                force_weight=original_config["force_weight"],
                stress_weight=original_config["stress_weight"],
            )
            for k, v in extra_val_metrics.items():
                logs[f"extra_val/{k}"] = v.mean().item()
            logs["extra_val/loss"] = extra_val_metrics["loss"].mean().item()
            wandb.log(logs, step=step)
            print(f"epoch {epoch:03d} extra val metrics: {logs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)
