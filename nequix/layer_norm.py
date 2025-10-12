from typing import Optional

import e3nn_jax as e3nn
import equinox as eqx
import jax
import jax.numpy as jnp


# based on https://github.com/facebookresearch/fairchem/blob/977a803/src/fairchem/core/models/esen/nn/layer_norm.py#L229
# from ESEN which is based on EquiformerV2
class RMSLayerNorm(eqx.Module):
    irreps: e3nn.Irreps = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    centering: bool = eqx.field(static=True)
    std_balance_degrees: bool = eqx.field(static=True)

    affine_weight: Optional[list[jax.Array]]
    affine_bias: Optional[list[jax.Array]]

    def __init__(
        self,
        irreps: e3nn.Irreps,
        eps: float = 1e-12,
        affine: bool = True,
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        self.irreps = e3nn.Irreps(irreps)
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        if self.affine:
            self.affine_weight = [jnp.ones(irr.mul) for irr in self.irreps]

            if self.centering:
                self.affine_bias = [
                    jnp.zeros(irr.mul) if irr.ir.l == 0 else None for irr in self.irreps
                ]
            else:
                self.affine_bias = None
        else:
            self.affine_weight = None
            self.affine_bias = None

    def __call__(self, node_input: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        input_chunks = []
        norms = []

        for i, (irr, x) in enumerate(zip(node_input.irreps, node_input.chunks)):
            if self.centering and irr.ir.l == 0:
                x = x - jnp.mean(x, axis=-2, keepdims=True)
            input_chunks.append(x)

            if self.std_balance_degrees:
                weight = 1 / (irr.ir.dim * len(self.irreps))
                # l2 norm
                norm = (x**2 * weight).sum(axis=-2, keepdims=True)
                # mean over channels
                norm = norm.mean(axis=-1, keepdims=True)
                norms.append(norm)
            else:
                raise NotImplementedError("not implemented")

        # sum across irreps (because we already weight by 1 / len(self.irreps))
        norm = jnp.concatenate(norms, axis=-1).sum(axis=-1, keepdims=True)
        norm = jnp.pow(norm + self.eps, -0.5)

        out_chunks = []
        for i, (irr, x) in enumerate(zip(node_input.irreps, input_chunks)):
            if self.affine_weight is not None:
                x = x * self.affine_weight[i][:, None]

            out = x * norm

            if self.affine_bias is not None and irr.ir.l == 0:
                out = out + self.affine_bias[i][:, None]

            out_chunks.append(out)

        return e3nn.from_chunks(node_input.irreps, out_chunks, node_input.shape[:-1])
