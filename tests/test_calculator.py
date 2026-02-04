import numpy as np
import pytest
import torch
import ase.build

from nequix.calculator import NequixCalculator
from nequix.model import OEQ_AVAILABLE

skip_no_oeq = pytest.mark.skipif(not OEQ_AVAILABLE, reason="OpenEquivariance not installed")


def si():
    return ase.build.bulk("Si", "diamond", a=5.43)


_atoms_relaxed = si()
_atoms_perturbed = si()
_atoms_perturbed.positions[0] += [0.1, 0.05, -0.05]


def f32(x):
    return np.array(x, dtype=np.float32)


REFERENCE_DATA = {
    "relaxed": {
        "atoms": _atoms_relaxed,
        "models": {
            "nequix-mp-1": {
                "energy": f32(-10.834069),
                "forces": f32(
                    [
                        [3.8649887e-08, 5.0524250e-08, -7.0780516e-08],
                        [-2.8871000e-08, -4.3772161e-08, 6.0237653e-08],
                    ]
                ),
                "stress": f32(
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
            "nequix-oam-1": {
                "energy": f32(-10.831113),
                "forces": f32(
                    [
                        [8.4808562e-08, 1.7345883e-08, 5.0291419e-08],
                        [-8.8242814e-08, -6.0535967e-09, -4.7482899e-08],
                    ]
                ),
                "stress": f32(
                    [
                        -1.8336197361e-02,
                        -1.8336204812e-02,
                        -1.8336208537e-02,
                        4.2584678006e-09,
                        2.3016142325e-09,
                        1.5990629931e-09,
                    ]
                ),
            },
        },
    },
    "perturbed": {
        "atoms": _atoms_perturbed,
        "models": {
            "nequix-mp-1": {
                "energy": f32(-10.753344),
                "forces": f32(
                    [[-1.1041114, -0.45131257, 0.45131272], [1.1041114, 0.45131263, -0.45131272]]
                ),
                "stress": f32(
                    [
                        -0.025367301,
                        -0.0278149154,
                        -0.0278149098,
                        -0.0241314191,
                        -0.0106064761,
                        0.0106064798,
                    ]
                ),
            },
            "nequix-oam-1": {
                "energy": f32(-10.765028),
                "forces": f32(
                    [[-0.930218, -0.36117458, 0.36117452], [0.930218, 0.36117464, -0.36117452]]
                ),
                "stress": f32(
                    [
                        -0.0198730454,
                        -0.0213395804,
                        -0.0213395748,
                        -0.020636579,
                        -0.0084719257,
                        0.0084719257,
                    ]
                ),
            },
        },
    },
}


@pytest.fixture(params=["relaxed", "perturbed"])
def structure(request):
    return request.param


@pytest.fixture
def atoms(structure):
    return REFERENCE_DATA[structure]["atoms"].copy()


@pytest.mark.parametrize("model_name", ["nequix-mp-1", "nequix-oam-1"])
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


@pytest.mark.parametrize(
    "backend, kernel",
    [
        ("torch", False),
        pytest.param("torch", True, marks=skip_no_oeq),
        ("jax", False),
        pytest.param("jax", True, marks=skip_no_oeq),
    ],
)
def test_calculator_nequix_mp_1_without_cell(backend, kernel):
    atoms = ase.build.molecule("H2O")
    calc = NequixCalculator(model_name="nequix-mp-1", backend=backend, use_kernel=kernel)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))
