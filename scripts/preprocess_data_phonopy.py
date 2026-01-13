import argparse
import multiprocessing
from pathlib import Path

import ase.db
import numpy as np
import phonopy
from ase.calculators.singlepoint import SinglePointCalculator
from phonopy.interface.phonopy_yaml import load_yaml
from tqdm import tqdm


def save_phonopy_yaml_to_ase_db(args):
    db_file, file_list, worker_id = args
    with ase.db.connect(db_file) as db:
        for file in tqdm(file_list, position=worker_id):
            ph_ref = phonopy.load(file)
            # NOTE: the MDR phonon files report unit cell energy, so we need to
            # multiply by the number of repetitions to get supercell energy
            energy_unitcell = load_yaml(file)["energy"]
            nrep = round(np.linalg.det(ph_ref.supercell_matrix))
            energy = energy_unitcell * nrep
            atoms = ase.Atoms(
                cell=ph_ref.supercell.cell,
                symbols=ph_ref.supercell.symbols,
                scaled_positions=ph_ref.supercell.scaled_positions,
                pbc=True,
            )
            ph_ref.produce_force_constants()
            hessian = (
                np.array(ph_ref.force_constants, dtype=np.float32)  # (n, n, 3, 3)
                .swapaxes(1, 2)  # (n, 3, n, 3)
                .reshape(3 * len(atoms), 3 * len(atoms))  # (3n, 3n)
            )
            # NOTE: MDR data does not include forces or stress, but the authors
            # report performing a relaxation to a convergence criterion of 1e-8
            # eV/A calculation, so we set forces and stress to zero
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=np.zeros_like(atoms.positions),
                stress=np.zeros_like(atoms.cell),
            )
            atoms.info["hessian"] = hessian
            db.write(atoms, data=atoms.info)


def preprocess(file_path, output_path, n_workers=16):
    file_path = Path(file_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if file_path.is_dir():
        file_paths = sorted(file_path.rglob("*.yaml.bz2"))
        chunks = np.array_split(file_paths, n_workers)
        db_files = [output_path / f"data_{i:04d}.aselmdb" for i in range(n_workers)]
        tasks = [(db_files[i], chunks[i], i) for i in range(n_workers) if len(chunks[i])]
        with multiprocessing.Pool(n_workers) as p:
            p.map(save_phonopy_yaml_to_ase_db, tasks)
    else:
        save_phonopy_yaml_to_ase_db((output_path / "data_0000.aselmdb", [file_path], 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, args.n_workers)


if __name__ == "__main__":
    main()
