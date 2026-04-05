#!/bin/bash
#SBATCH --account=ai4dd
#SBATCH --partition=gpu
#SBATCH --cpus-per-gpu=15
#SBATCH --mem=0
#SBATCH --time=7-0:0:0
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=nequix_train
#SBATCH --output=train_%j.log

source /cv/home/daigavaa/nequix/.venv/bin/activate

# nequix_train configs/nequix-omol-neutral.yml
# nequix_train configs/nequix-omol-neutral-repulsion.yml
# nequix_train configs/nequix-omol-neutral-index.yml
nequix_train configs/nequix-omol-neutral-index-repulsion.yml
