#!/bin/bash
#SBATCH --account=ai4dd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=0
#SBATCH --time=7-0:0:0
#SBATCH --job-name=nequix_train
#SBATCH --output=train_%j.log

source /cv/home/daigavaa/nequix/.venv/bin/activate
export LD_LIBRARY_PATH=/cv/home/daigavaa/nequix/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib

nequix_train configs/nequix-omol-neutral.yml
