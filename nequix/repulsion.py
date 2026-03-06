import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path


def make_NLH_repulsion(
    atomic_numbers: np.ndarray,
    coefficients_file: str = Path(__file__).parent / "nlh_coeffs.dat",
    alchemical_softcore_alpha: float = 0.5,
    alchemical_m: float = 2.0,
):
    """Creates a repulsion function (in eV) for the NLH potential."""
    atomic_numbers = jnp.array(atomic_numbers, dtype=jnp.int32)

    DATA_NLH = np.loadtxt(coefficients_file, usecols=np.arange(0, 8))
    zmax = int(np.max(DATA_NLH[:, 0]))
    AB = np.zeros(((zmax + 1) ** 2, 6), dtype=np.float32)
    for i in range(DATA_NLH.shape[0]):
        z1 = int(DATA_NLH[i, 0])
        z2 = int(DATA_NLH[i, 1])
        AB[z1 + zmax * z2] = DATA_NLH[i, 2:8]
        AB[z2 + zmax * z1] = DATA_NLH[i, 2:8]
    AB = AB.reshape((zmax + 1) ** 2, 3, 2)

    CS = jnp.array(AB[:, :, 0])  # (zmax+1)^2, 3
    ALPHAS = jnp.array(AB[:, :, 1])  # (zmax+1)^2, 3

    BOHR_TO_ANG = 0.52917721
    HARTREE_TO_EV = 27.211386

    def repulsion_fn(
        species: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        rijs: jax.Array,
        cutoffs: jax.Array,
        alchemical_group: jax.Array | None = None,
        alchemical_lambda: float | None = None,
    ) -> jax.Array:
        Z = atomic_numbers[species]
        s12 = Z[senders] + zmax * Z[receivers]
        cs = CS[s12]
        alphas = ALPHAS[s12]

        if alchemical_group is not None:
            same_alchemical_group = (
                alchemical_group[senders] == alchemical_group[receivers]
            )
            alch_alpha = alchemical_softcore_alpha**2 * (1 - alchemical_lambda)
            rijs = jnp.where(
                same_alchemical_group, rijs, jnp.sqrt(rijs**2 + alch_alpha)
            )
            lambda_v = 0.5 * (1 - jnp.cos(jnp.pi * alchemical_lambda))
            cutoffs = jnp.where(
                same_alchemical_group, cutoffs, (lambda_v**alchemical_m) * cutoffs
            )

        Z = jnp.where(Z > 0, Z.astype(rijs.dtype), 0.0)
        phi = (cs * jnp.exp(-alphas * rijs[:, None])).sum(axis=-1)
        Zij = Z[senders] * Z[receivers] * cutoffs
        E_rep_pair = Zij * phi / rijs
        E_rep = jnp.zeros(Z.shape[0]).at[senders].add(E_rep_pair)
        E_rep *= 0.5 * BOHR_TO_ANG * HARTREE_TO_EV
        return E_rep

    return repulsion_fn
