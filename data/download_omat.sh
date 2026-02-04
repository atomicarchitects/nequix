#!/bin/bash

# see https://huggingface.co/datasets/facebook/OMAT24/blob/main/README.md for all files 

# configure data root as first arg or DATA_ROOT env var (defaults to ./data)
DATA_ROOT="${1:-${DATA_ROOT:-data}}"

# train files
mkdir -p "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-1000.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-1000-subsampled.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-1000-subsampled.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-500.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500-subsampled.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-500-subsampled.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-300.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-300-subsampled.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-300-subsampled.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-npt.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/aimd-from-PBE-1000-npt.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-1000-nvt.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/aimd-from-PBE-1000-nvt.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-npt.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/aimd-from-PBE-3000-npt.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/aimd-from-PBE-3000-nvt.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/aimd-from-PBE-3000-nvt.tar.gz" -C "$DATA_ROOT/omat/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-relax.tar.gz -P "$DATA_ROOT/omat/train/"
tar -xf "$DATA_ROOT/omat/train/rattled-relax.tar.gz" -C "$DATA_ROOT/omat/train/"

# val files
mkdir -p "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/rattled-1000.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-1000-subsampled.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/rattled-1000-subsampled.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/rattled-500.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500-subsampled.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/rattled-500-subsampled.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/rattled-300.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-300-subsampled.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/rattled-300-subsampled.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-npt.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/aimd-from-PBE-1000-npt.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-1000-nvt.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/aimd-from-PBE-1000-nvt.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-npt.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/aimd-from-PBE-3000-npt.tar.gz" -C "$DATA_ROOT/omat/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/aimd-from-PBE-3000-nvt.tar.gz -P "$DATA_ROOT/omat/val/"
tar -xf "$DATA_ROOT/omat/val/aimd-from-PBE-3000-nvt.tar.gz" -C "$DATA_ROOT/omat/val/"

# sAlex train files
mkdir -p "$DATA_ROOT/salex/train/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/train.tar.gz -P "$DATA_ROOT/salex/train/"
tar -xf "$DATA_ROOT/salex/train/train.tar.gz" -C "$DATA_ROOT/salex/train/"

# sAlex val files
mkdir -p "$DATA_ROOT/salex/val/"
wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/sAlex/val.tar.gz -P "$DATA_ROOT/salex/val/"
tar -xf "$DATA_ROOT/salex/val/val.tar.gz" -C "$DATA_ROOT/salex/val/"


