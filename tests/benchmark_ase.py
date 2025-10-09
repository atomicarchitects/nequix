import ase
from ase.build import bulk
import time
from nequix.calculator import NequixCalculator

def benchmark(size, calc):
    atoms = ase.build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    print("Number of atoms: ", len(atoms))
    atoms.calc = calc

    start = time.time()
    for _ in range(10):
        E = atoms.get_potential_energy()
        atoms.rattle()
    end = time.time()

    start = time.time()
    for _ in range(100):
        E = atoms.get_potential_energy()
        # print(E)
        atoms.rattle()
    end = time.time()
    return (end - start) / 100


calculator_nequix = NequixCalculator(
    model_path="./models/nequix-mp-1.nqx",
)
for size in [1, 1, 2, 3, 4, 5, 6, 7]:
    print("Nequix: ", benchmark(size, calculator_nequix) * 1000, "ms")
