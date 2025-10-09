# Modified from https://github.com/KellerJordan/Muon/blob/f90a42b/muon.py
# to work with e3nn Linear

import torch
import torch.distributed as dist


def zeropower_via_newtonschulz5(G, steps: int):
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, slices_e3nn_linear=None):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)

    if slices_e3nn_linear is not None:
        update_list = []
        for index_slice, shape_2D in slices_e3nn_linear:  # square weight slices of updates
            weight_slice = update[index_slice].reshape(shape_2D)
            grad_slice = grad[index_slice].reshape(shape_2D)
            update_weight_slice = zeropower_via_newtonschulz5(weight_slice, steps=ns_steps)
            update_weight_slice *= max(1, grad_slice.size(-2) / grad_slice.size(-1)) ** 0.5
            # Flatten the weight slice
            update_list.append(update_weight_slice.flatten())
        # concatenate updates back into single vector
        update = torch.cat(update_list, dim=-1)
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert (
            isinstance(params, list)
            and len(params) >= 1
            and isinstance(params[0], torch.nn.Parameter)
        )
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)

                if "slices_e3nn_linear" in group:
                    assert set(group.keys()) == set(
                        [
                            "params",
                            "lr",
                            "momentum",
                            "weight_decay",
                            "use_muon",
                            "slices_e3nn_linear",
                        ]
                    )
                else:
                    assert set(group.keys()) == set(
                        ["params", "lr", "momentum", "weight_decay", "use_muon"]
                    )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                if "slices_e3nn_linear" in group:
                    assert set(group.keys()) == set(
                        [
                            "params",
                            "lr",
                            "betas",
                            "eps",
                            "weight_decay",
                            "use_muon",
                            "slices_e3nn_linear",
                        ]
                    )
                else:
                    assert set(group.keys()) == set(
                        ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                    )
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for i, p in enumerate(group["params"]):
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if "slices_e3nn_linear" in group:
                        # Need to send in the slices per linear
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            slices_e3nn_linear=group["slices_e3nn_linear"][i],
                        )
                    else:
                        update = muon_update(
                            p.grad, state["momentum_buffer"], beta=group["momentum"]
                        )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
