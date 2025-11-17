import multiprocessing
import argparse
from pathlib import Path

import ase.db
import ase.io
import numpy as np
from tqdm import tqdm


def save_atoms_to_ase_db(args):
    db_file, file_list, worker_id = args
    with ase.db.connect(db_file) as db:
        for file in tqdm(file_list, position=worker_id):
            atoms_list = ase.io.read(file, index=":")
            for atoms in atoms_list:
                db.write(atoms, data=atoms.info)

def preprocess(file_path, output_path, n_workers=16):
    file_path = Path(file_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if file_path.is_dir():
        file_paths = sorted(file_path.rglob("*.extxyz"))
        chunks = np.array_split(file_paths, n_workers)
        db_files = [output_path / f"data_{i:04d}.aselmdb" for i in range(n_workers)]
        tasks = [(db_files[i], chunks[i], i) for i in range(n_workers) if len(chunks[i])]
        with multiprocessing.Pool(n_workers) as p:
            p.map(save_atoms_to_ase_db, tasks)
    else:
        save_atoms_to_ase_db((output_path / "data_0000.aselmdb", [file_path], 0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--n_workers", type=int, default=16)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path, args.n_workers)

if __name__ == "__main__":
    main()
