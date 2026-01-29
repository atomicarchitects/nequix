import argparse
import copy
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from nequix.data import AseDBDataset, ConcatDataset
from nequix.torch.model import (
    NequixTorch,
    get_optimizer_param_groups,
    load_model,
    save_model,
    scatter,
)
from nequix.torch.utils import StatefulDistributedSampler


def loss(model, batch, energy_weight, force_weight, stress_weight, loss_type="huber", device="cpu"):
    """Return huber loss and MAE of energy and force in eV and eV/Å respectively"""
    energy_per_atom, forces, stress = model(
        batch.x,
        batch.positions,
        batch.edge_attr,
        batch.edge_index,
        batch.cell if hasattr(batch, "cell") else None,
        batch.n_node,
        batch.n_edge,
        batch.batch,
    )
    energy = scatter(energy_per_atom, batch.batch, dim=0, dim_size=batch.n_node.size(0))
    config = {
        "mse": {"energy": "mse", "force": "mse", "stress": "mse"},
        "huber": {"energy": "huber", "force": "huber", "stress": "huber"},
        "mae": {"energy": "mae", "force": "l2", "stress": "mae"},
    }[loss_type]

    loss_fns = {
        "mae": lambda pred, true: torch.abs(pred - true),
        "mse": lambda pred, true: (pred - true) ** 2,
        "huber": lambda pred, true: nn.functional.huber_loss(
            pred, true, delta=0.1, reduction="mean"
        ),
    }

    num_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
    energy_loss_per_atom = (
        torch.sum(loss_fns[config["energy"]](energy / batch.n_node, batch["energy"] / batch.n_node))
        / num_graphs
    )

    if config["force"] == "l2":
        force_vector_norm = torch.linalg.vector_norm(forces - batch.forces, ord=2, dim=-1)
        force_loss = torch.sum(force_vector_norm) / batch.x.size(0)
    else:
        force_loss = torch.sum(loss_fns[config["force"]](forces, batch.forces)) / (
            3 * batch.x.size(0)
        )

    if stress_weight > 0 and hasattr(batch, "stress"):
        stress_loss = torch.sum(loss_fns[config["stress"]](stress, batch.stress)) / (9 * num_graphs)
    else:
        stress_loss = 0

    total_loss = (
        energy_weight * energy_loss_per_atom
        + force_weight * force_loss
        + stress_weight * stress_loss
    )

    # metrics:

    # MAE energy
    energy_mae_per_atom = (
        torch.sum(torch.abs(energy / batch.n_node - batch.energy / batch.n_node)) / num_graphs
    )

    # MAE forces
    force_mae = torch.sum(torch.abs(forces - batch.forces)) / (3 * batch.x.size(0))

    # MAE stress
    if hasattr(batch, "stress"):
        stress_mae_per_atom = torch.sum(
            torch.abs(stress - batch.stress) / batch.n_node[:, None, None]
        ) / (9 * num_graphs)
    else:
        stress_mae_per_atom = torch.tensor(0.0, device=device)

    return total_loss, {
        "energy_mae_per_atom": energy_mae_per_atom,
        "force_mae": force_mae,
        "stress_mae_per_atom": stress_mae_per_atom,
    }


def evaluate(
    model,
    dataloader,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=1.0,
    loss_type="huber",
    device="cpu",
):
    """Return loss and RMSE of energy and force in eV and eV/Å respectively"""
    model.eval()
    total_metrics = defaultdict(float)
    total_count = 0

    for batch in dataloader:
        batch = batch.to(device)
        n_graphs = batch.num_graphs
        val_loss, metrics = loss(
            model, batch, energy_weight, force_weight, stress_weight, loss_type, device
        )
        total_metrics["loss"] += val_loss.item() * n_graphs
        for key, value in metrics.items():
            total_metrics[key] += value.item() * n_graphs
        total_count += n_graphs

    for key, value in total_metrics.items():
        total_metrics[key] = value / total_count

    return total_metrics


def save_training_state(
    path,
    model,
    ema_model,
    optimizer,
    scheduler,
    global_step,
    steps_through_epoch,
    epoch,
    best_val_loss,
    wandb_run_id=None,
):
    # Extract state dict from DDP wrapper if needed
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    ema_state = (
        ema_model.module.state_dict() if hasattr(ema_model, "module") else ema_model.state_dict()
    )

    state = {
        "model_state_dict": model_state,
        "ema_model_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "steps_through_epoch": steps_through_epoch,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb_run_id,
    }
    torch.save(state, path)


def load_training_state(path, model, ema_model, optimizer, scheduler):
    state = torch.load(path, map_location="cpu", weights_only=False)

    # Load state dicts into the correct model (handle DDP wrapper)
    if hasattr(model, "module"):
        model.module.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state["model_state_dict"])

    if hasattr(ema_model, "module"):
        ema_model.module.load_state_dict(state["ema_model_state_dict"])
    else:
        ema_model.load_state_dict(state["ema_model_state_dict"])

    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])

    return (
        model,
        ema_model,
        optimizer,
        scheduler,
        state["global_step"],
        state["steps_through_epoch"],
        state["epoch"],
        state["best_val_loss"],
        state.get("wandb_run_id"),
    )


def train(config_path: str):
    """Train a Nequack model from a config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # use TMPDIR for slurm jobs if available
    config["cache_dir"] = config.get("cache_dir") or os.environ.get("TMPDIR")

    # Distributed training setup
    is_distributed = "RANK" in os.environ
    if is_distributed:
        setup_ddp()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = torch.device(f"cuda:{local_rank}")
        print(f"Rank {rank}/{world_size}: Using device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        world_size = 1
        print(f"Using device: {device}")

    if isinstance(config["train_path"], list):
        train_dataset = ConcatDataset(
            [
                AseDBDataset(
                    file_path=path,
                    atomic_numbers=config["atomic_numbers"],
                    cutoff=config["cutoff"],
                    backend="torch",
                )
                for path in config["train_path"]
            ]
        )
    else:
        train_dataset = AseDBDataset(
            file_path=config["train_path"],
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
            backend="torch",
        )
    if "valid_frac" in config:
        train_dataset, val_dataset = train_dataset.split(valid_frac=config["valid_frac"])
    else:
        assert "valid_path" in config, "valid_path must be specified if valid_frac is not provided"
        val_dataset = AseDBDataset(
            file_path=config["valid_path"],
            atomic_numbers=config["atomic_numbers"],
            cutoff=config["cutoff"],
            backend="torch",
        )

    if "atom_energies" in config:
        atom_energies = [config["atom_energies"][n] for n in config["atomic_numbers"]]
    else:
        raise NotImplementedError("average atom energies not implemented for torch backend")
        # atom_energies = average_atom_energies(train_dataset)

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
        raise NotImplementedError("dataset stats not implemented for torch backend")
        # stats = dataset_stats(train_dataset, atom_energies)

    if is_distributed:
        train_sampler = StatefulDistributedSampler(
            train_dataset,
            batch_size=config["batch_size"],
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        val_sampler = StatefulDistributedSampler(
            val_dataset,
            batch_size=config["batch_size"],
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42,
        )

        # Note: We assume 16 cores per task
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            sampler=train_sampler,
            num_workers=16,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            sampler=val_sampler,
            num_workers=16,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=16,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

    # Set the seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

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
        shift=stats["shift"],
        scale=stats["scale"],
        avg_n_neighbors=stats["avg_n_neighbors"],
        atom_energies=atom_energies,
        kernel=config["kernel"],
    ).to(device)

    if "finetune_from" in config and Path(config["finetune_from"]).exists():
        model, _ = load_model(config["finetune_from"])
        if "atom_energies" in config:
            model.atom_energies = torch.tensor(
                [config["atom_energies"][n] for n in config["atomic_numbers"]],
                dtype=torch.float64,
            )
        if "scale" in config:
            model.scale = torch.tensor(config["scale"])
        if "shift" in config:
            model.shift = torch.tensor(config["shift"])
        model.to(device)

    if rank == 0:
        print(model)
        param_count = sum(p.numel() for p in model.parameters())

    steps_per_epoch = len(train_dataset) // (config["batch_size"] * world_size)
    total_steps = config["n_epochs"] * steps_per_epoch
    warmup_steps = config["warmup_epochs"] * steps_per_epoch

    if config["optimizer"] == "muon":
        weights_2d = [p for p in model.parameters() if p.ndim >= 2]
        weights_layer_norm = []
        weights_e3nn_linear = []
        slices_e3nn_linear = []
        for layer in model.layers:
            weights_layer_norm.extend([weight for weight in layer.layer_norm.affine_weight])
            if layer.layer_norm.affine_bias is not None:
                weights_layer_norm.extend([weight for weight in layer.layer_norm.affine_bias])
            weights_e3nn_linear.extend(
                [layer.linear_1.weight, layer.linear_2.weight, layer.skip.weight]
            )
            slices_e3nn_linear.extend(
                [
                    layer.linear_1.weight_index_slices,
                    layer.linear_2.weight_index_slices,
                    layer.skip.weight_index_slices,
                ]
            )

        param_groups = [
            # Pass in additional slicing metadata. For each linear weight we have a list of irreps that slice it.
            {
                "params": weights_e3nn_linear,
                "slices_e3nn_linear": slices_e3nn_linear,
                "use_muon": True,
                "lr": config["learning_rate"],
                "weight_decay": config["weight_decay"],
            },
            {
                "params": weights_2d,
                "use_muon": True,
                "lr": config["learning_rate"],
                "weight_decay": config["weight_decay"],
            },
            # LayerNorm should use AdamW optimizer
            {
                "params": weights_layer_norm,
                "use_muon": False,
                "lr": config["learning_rate"],
                "weight_decay": 0.0,
            },
        ]
        from nequix.torch.muon import SingleDeviceMuonWithAuxAdam

        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    else:
        param_groups = get_optimizer_param_groups(model, config["weight_decay"])
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config["learning_rate"],
        )

    # EMA model - deep copy of the original model
    ema_model = copy.deepcopy(model).to(device)

    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Learning rate scheduler with warmup and cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return config["warmup_factor"] + (1 - config["warmup_factor"]) * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize step and checkpoint loading variables
    global_step = 0
    checkpoint_steps_through_epoch = 0
    start_epoch = 0
    best_val_loss = float("inf")
    wandb_run_id = None
    wandb_sync = lambda: None  # noqa: E731

    # Load checkpoint if resuming and checkpoint exists
    if "resume_from" in config and Path(config["resume_from"]).exists():
        (
            model,
            ema_model,
            optimizer,
            scheduler,
            global_step,
            checkpoint_steps_through_epoch,
            start_epoch,
            best_val_loss,
            wandb_run_id,
        ) = load_training_state(config["resume_from"], model, ema_model, optimizer, scheduler)

    # Only initialize wandb on rank 0
    if rank == 0:
        wandb_sync = (
            TriggerWandbSyncHook() if os.environ.get("WANDB_MODE") == "offline" else lambda: None
        )

        wandb_init_kwargs = {"project": "nequix", "config": config}
        if wandb_run_id:
            wandb_init_kwargs.update({"id": wandb_run_id, "resume": "allow"})

        wandb.init(**wandb_init_kwargs)

        if hasattr(wandb, "run") and wandb.run is not None:
            wandb.run.summary["param_count"] = param_count
            wandb_run_id = getattr(wandb.run, "id", None)

    def train_step(model, ema_model, batch, step):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        total_loss, metrics = loss(
            model,
            batch,
            config["energy_weight"],
            config["force_weight"],
            config["stress_weight"],
            config["loss_type"],
            device,
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if "grad_clip_norm" in config:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        metrics["grad_norm"] = grad_norm
        optimizer.step()
        scheduler.step()

        # Update EMA model
        # don't weight early steps as much (from https://github.com/fadel/pytorch_ema)
        decay = min(config.get("ema_decay", 0.999), (1 + step) / (10 + step))

        with torch.no_grad():
            # Handle DDP wrapper for parameter access
            model_params = (
                model.module.parameters() if hasattr(model, "module") else model.parameters()
            )
            ema_params = (
                ema_model.module.parameters()
                if hasattr(ema_model, "module")
                else ema_model.parameters()
            )
            for ema_param, model_param in zip(ema_params, model_params):
                ema_param.data = ema_param.data * decay + model_param.data * (1 - decay)

        return total_loss, metrics

    if "resume_from" in config and checkpoint_steps_through_epoch > 0:
        train_loader.sampler.set_start_iter(checkpoint_steps_through_epoch)

    for epoch in range(start_epoch, config["n_epochs"]):
        # Set epoch for distributed sampler
        if is_distributed:
            train_loader.sampler.set_epoch(epoch)

        start_time = time.time()

        for steps_through_epoch, batch in enumerate(
            train_loader, start=checkpoint_steps_through_epoch
        ):
            batch_time = time.time() - start_time
            start_time = time.time()

            batch = batch.to(device)

            total_loss, metrics = train_step(model, ema_model, batch, global_step)

            train_time = time.time() - start_time
            global_step += 1

            if global_step % config["log_every"] == 0 and rank == 0:
                logs = {}
                logs["train/loss"] = total_loss.item()
                logs["learning_rate"] = scheduler.get_last_lr()[0]
                logs["train/batch_time"] = batch_time
                logs["train/train_time"] = train_time
                for key, value in metrics.items():
                    logs[f"train/{key}"] = value.item()
                logs["train/batch_size"] = batch.num_graphs if hasattr(batch, "num_graphs") else 1
                wandb.log(logs, step=global_step)
                print(f"step: {global_step}, logs: {logs}")
                wandb_sync()

                save_training_state(
                    Path(wandb.run.dir) / "state.pkl",
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    global_step,
                    steps_through_epoch,
                    epoch,
                    best_val_loss,
                    wandb_run_id=wandb_run_id,
                )

                if "state_path" in config:
                    save_training_state(
                        config["state_path"],
                        model,
                        ema_model,
                        optimizer,
                        scheduler,
                        global_step,
                        steps_through_epoch,
                        epoch,
                        best_val_loss,
                        wandb_run_id=wandb_run_id,
                    )

            start_time = time.time()

        # reset sampler to start of epoch
        train_loader.sampler.set_start_iter(0)
        checkpoint_steps_through_epoch = 0

        if rank == 0:
            val_metrics = evaluate(
                ema_model,
                val_loader,
                config["energy_weight"],
                config["force_weight"],
                config["stress_weight"],
                config["loss_type"],
                device,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                model_to_save = ema_model.module if hasattr(ema_model, "module") else ema_model
                if hasattr(wandb, "run") and wandb.run is not None:
                    save_model(Path(wandb.run.dir) / "checkpoint.pt", model_to_save, config)

            logs = {}
            for key, value in val_metrics.items():
                logs[f"val/{key}"] = value
            logs["epoch"] = epoch
            if hasattr(wandb, "run") and wandb.run is not None:
                wandb.log(logs, step=global_step)
            print(f"epoch: {epoch}, logs: {logs}")
            wandb_sync()

    if is_distributed and epoch == config["n_epochs"] - 1:
        cleanup_ddp()


def setup_ddp():
    """Initialize distributed training"""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Clean up distributed training"""
    destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()
    train(args.config_path)


if __name__ == "__main__":
    main()
