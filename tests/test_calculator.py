import numpy as np
import pytest
import torch
import ase.build

from nequix.calculator import NequixCalculator


def si():
    return ase.build.bulk("Si", "diamond", a=5.43)


_atoms_relaxed = si()
_atoms_perturbed = si()
_atoms_perturbed.positions[0] += [0.1, 0.05, -0.05]


array = lambda x: np.array(x, dtype=np.float32)
REFERENCE_DATA = {
    "relaxed": {
        "atoms": _atoms_relaxed,
        "models": {
            "nequix-mp-1": {
                "energy": array(-10.834069),
                "forces": array(
                    [
                        [3.8649887e-08, 5.0524250e-08, -7.0780516e-08],
                        [-2.8871000e-08, -4.3772161e-08, 6.0237653e-08],
                    ]
                ),
                "stress": array(
                    [
                        -2.6995424181e-02,
                        -2.6995424181e-02,
                        -2.6995424181e-02,
                        6.4219904949e-09,
                        -1.2764893587e-09,
                        -2.4910233876e-09,
                    ]
                ),
            },
        },
    },
    "perturbed": {
        "atoms": _atoms_perturbed,
        "models": {
            "nequix-mp-1": {
                "energy": array(-10.753344),
                "forces": array(
                    [[-1.1041114, -0.45131257, 0.45131272], [1.1041114, 0.45131263, -0.45131272]]
                ),
                "stress": array(
                    [
                        -0.025367301,
                        -0.0278149154,
                        -0.0278149098,
                        -0.0241314191,
                        -0.0106064761,
                        0.0106064798,
                    ]
                ),
            }
        },
    },
}

@pytest.fixture(params=["relaxed", "perturbed"])
def structure(request):
    return request.param


@pytest.fixture
def atoms(structure):
    return REFERENCE_DATA[structure]["atoms"].copy()


@pytest.mark.parametrize("model_name", ["nequix-mp-1"])
@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("use_kernel", [True, False])
def test_nequix_calculator_matches_reference(structure, atoms, model_name, backend, use_kernel):
    if backend == "torch" and use_kernel and not torch.cuda.is_available():
        pytest.skip("Torch kernel requires CUDA")

    if backend == "jax" and use_kernel:
        pytest.skip("No kernel support for JAX")

    reference = REFERENCE_DATA[structure]["models"][model_name]

    atoms.calc = NequixCalculator(model_name=model_name, backend=backend, use_kernel=use_kernel)

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=True)

    np.testing.assert_allclose(energy, reference["energy"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(forces, reference["forces"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(stress, reference["stress"], atol=1e-5, rtol=1e-5)
