import json
import argparse
from pathlib import Path

# import shutil
import os

import numpy as np

N_TEST = 1000  # number of hold out ids for testing
SEED = 42

# see scripts/check_mdr_compliance.py. these are the ids that are not in the
# mptrj dataset
NOT_IN_MPTRJ = {
    "mp-7339",
    "mp-867943",
    "mp-555153",
    "mp-626902",
    "mp-552806",
    "mp-3887",
    "mp-867286",
    "mp-867300",
    "mp-867712",
    "mp-867322",
    "mp-867929",
    "mp-625416",
    "mp-23151",
    "mp-6643",
    "mp-867371",
    "mp-771888",
    "mp-867203",
    "mp-867700",
    "mp-768573",
    "mp-978857",
    "mp-42160",
    "mp-867168",
    "mp-625052",
    "mp-867570",
    "mp-867133",
    "mp-867329",
    "mp-12992",
    "mp-3033",
    "mp-867674",
    "mp-867261",
    "mp-555065",
    "mp-867192",
    "mp-632394",
    "mp-561394",
    "mp-4748",
    "mp-772812",
    "mp-646925",
    "mp-569349",
    "mp-867577",
    "mp-867528",
    "mp-867607",
    "mp-553973",
    "mp-867129",
    "mp-557452",
    "mp-613442",
    "mp-754368",
    "mp-867194",
    "mp-867334",
    "mp-6799",
    "mp-643770",
    "mp-556062",
    "mp-867699",
    "mp-656662",
    "mp-769396",
    "mp-619456",
    "mp-614491",
    "mp-561623",
    "mp-867964",
    "mp-867953",
    "mp-684708",
    "mp-558347",
    "mp-867355",
    "mp-867328",
    "mp-867730",
    "mp-626369",
    "mp-753541",
    "mp-8426",
    "mp-626557",
    "mp-625917",
    "mp-627016",
    "mp-11695",
    "mp-558023",
    "mp-867757",
    "mp-6356",
    "mp-654956",
    "mp-867884",
    "mp-625837",
    "mp-769284",
    "mp-6929",
    "mp-867132",
    "mp-867839",
    "mp-626878",
    "mp-867615",
    "mp-561196",
    "mp-625678",
    "mp-867335",
    "mp-6425",
    "mp-625056",
    "mp-616190",
    "mp-754668",
    "mp-4122",
}


def symlink_file(src, dst, use_relative=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    target = Path(os.path.relpath(src, start=dst.parent)) if use_relative else src.resolve()
    dst.symlink_to(target)


def main(data_root: Path):
    input_dir = data_root / "phonon_benchmark/pbe"
    input_dir_disp = data_root / "phonon_benchmark/disp_struct"
    out_dir = data_root / "mdr-pbe/"
    out_dir_disp = data_root / "mdr-pbe-disp/"

    files = sorted(list(input_dir.glob("*.yaml.bz2")))

    n_holdout = N_TEST
    extra_holdout = True
    split_file = data_root / "pbe_mdr_split.json"
    ids_train, ids_val, ids_test = generate_split(files, n_holdout, split_file=split_file)
    print("n_holdout:", n_holdout)
    print("extra_holdout:", extra_holdout)
    print("ids_train:", len(ids_train))
    print("ids_val:", len(ids_val))
    print("ids_test:", len(ids_test))
    out_dir.mkdir(parents=True, exist_ok=True)
    # copy phonon yaml files
    (out_dir / "test").mkdir(parents=True, exist_ok=True)
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    for id in ids_test:
        # shutil.copy(INPUT_DIR / f"{id}.yaml.bz2", out_dir / "test" / f"{id}.yaml.bz2")
        symlink_file(input_dir / f"{id}.yaml.bz2", out_dir / "test" / f"{id}.yaml.bz2")
    for id in ids_train:
        # shutil.copy(INPUT_DIR / f"{id}.yaml.bz2", out_dir / "train" / f"{id}.yaml.bz2")
        symlink_file(input_dir / f"{id}.yaml.bz2", out_dir / "train" / f"{id}.yaml.bz2")
    for id in ids_val:
        # shutil.copy(INPUT_DIR / f"{id}.yaml.bz2", out_dir / "val" / f"{id}.yaml.bz2")
        symlink_file(input_dir / f"{id}.yaml.bz2", out_dir / "val" / f"{id}.yaml.bz2")

    # copy displacement extxyz files
    for id in ids_test:
        (out_dir_disp / "test" / id).mkdir(parents=True, exist_ok=True)
        for file in (input_dir_disp / id).glob("*.extxyz"):
            # shutil.copy(file, out_dir_disp / "test" / id / file.name)
            symlink_file(file, out_dir_disp / "test" / id / file.name)
    for id in ids_train:
        (out_dir_disp / "train" / id).mkdir(parents=True, exist_ok=True)
        for file in (input_dir_disp / id).glob("*.extxyz"):
            # shutil.copy(file, out_dir_disp / "train" / id / file.name)
            symlink_file(file, out_dir_disp / "train" / id / file.name)
    for id in ids_val:
        (out_dir_disp / "val" / id).mkdir(parents=True, exist_ok=True)
        for file in (input_dir_disp / id).glob("*.extxyz"):
            # shutil.copy(file, out_dir_disp / "val" / id / file.name)
            symlink_file(file, out_dir_disp / "val" / id / file.name)


def generate_split(files, n_holdout, split_file, val_frac=0.05):
    if split_file.exists():
        with open(split_file, "r") as f:
            split = json.load(f)
        return split["train"], split["val"], split["test"]

    ids = [file.name.removesuffix(".yaml.bz2") for file in files]
    ids = set(ids)

    rng = np.random.RandomState(seed=SEED)
    # test ids are N_TEST ids including all the ones not in mptrj
    if n_holdout > len(NOT_IN_MPTRJ):
        ids_test = (
            set(
                rng.choice(list(ids - NOT_IN_MPTRJ), size=N_TEST - len(NOT_IN_MPTRJ), replace=False)
            )
            ^ NOT_IN_MPTRJ
        )
    else:
        ids_test = NOT_IN_MPTRJ

    # train ids are all the ids minus the test ids (so no overlap, and only includes mptrj ids)
    ids_train_val = ids - ids_test
    ids_train = set(
        rng.choice(
            list(ids_train_val), size=int(len(ids_train_val) * (1 - val_frac)), replace=False
        )
    )
    ids_val = ids_train_val - ids_train
    split = {
        "train": list(ids_train),
        "val": list(ids_val),
        "test": list(ids_test),
    }
    with open(split_file, "w") as f:
        json.dump(split, f)
    return ids_train, ids_val, ids_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    args = parser.parse_args()
    main(Path(args.data_root))
