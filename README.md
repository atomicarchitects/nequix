<h1 align='center'>Nequix</h1>

Source code and model weights for the [Nequix foundation model](https://arxiv.org/abs/2508.16067), and [Phonon fine-tuning (PFT)](https://arxiv.org/abs/2601.07742).

Model | Dataset | Theory | Reference
---   | ---     | ---    | ---
`nequix-mp-1`| MPtrj | DFT (PBE+U) | [Nequix](https://arxiv.org/abs/2508.16067)
`nequix-mp-1-pft`| MPtrj, MDR Phonon | DFT (PBE+U) |[PFT](https://arxiv.org/abs/2601.07742)
`nequix-omat-1`| OMat24 | DFT (PBE+U, VASP 54) | [PFT](https://arxiv.org/abs/2601.07742)
`nequix-oam-1`| OMat24, sAlex, MPtrj | DFT (PBE+U) | [PFT](https://arxiv.org/abs/2601.07742)
`nequix-oam-1-pft`| OMat24, sAlex, MPtrj, MDR Phonon | DFT (PBE+U) | [PFT](https://arxiv.org/abs/2601.07742)


## Usage

### Installation

```bash
pip install nequix
```

or for torch

```bash
pip install nequix[torch]
```

### ASE calculator

Using `nequix.calculator.NequixCalculator`, you can perform calculations in
ASE with a pre-trained Nequix model.

```python
from nequix.calculator import NequixCalculator

atoms = ...
atoms.calc = NequixCalculator("nequix-mp-1", backend="jax")
```

or if you want to use the faster PyTorch + kernels backend

```python
...
atoms.calc = NequixCalculator("nequix-mp-1", backend="torch")
...
```

#### NequixCalculator

Arguments
- `model_name` (str, default "nequix-mp-1"): Pretrained model alias to load or download.
- `model_path` (str | Path, optional): Path to local checkpoint; overrides `model_name`.
- `backend` ({"jax", "torch"}, default "jax"): Compute backend.
- `capacity_multiplier` (float, default 1.1): JAX-only; padding factor to limit recompiles.
- `use_compile` (bool, default True): Torch-only; on GPU, uses `torch.compile()`.
- `use_kernel` (bool, default True): Torch-only; on GPU, use [OpenEquivariance](https://github.com/PASSIONLab/OpenEquivariance) kernels.

### Training

Models are trained with the `nequix_train` command using a single `.yml`
configuration file:

```bash
nequix_train <config>.yml
```
or for Torch

```bash
# Single GPU
uv sync --extra torch
uv run nequix/torch/train.py <config>.yml
# Multi-GPU
uv run torchrun --nproc_per_node=<gpus> nequix/torch/train.py <config>.yml
```

To reproduce the training of Nequix-MP-1, first clone the repo and sync the environment:

```bash
git clone https://github.com/atomicarchitects/nequix.git
cd nequix
uv sync
```


Then download the MPtrj data from
https://figshare.com/files/43302033 into `data/` then run the following to extract the data:

```bash
bash data/download_mptrj.sh
```

Preprocess the data into `.aselmdb` files:

```bash
uv run scripts/preprocess_data.py data/mptrj-gga-ggapu data/mptrj-aselmdb
```

Then start the training run:
```bash
nequix_train configs/nequix-mp-1.yml
```

This will take less than 125 hours on a single 4 x A100 node (<25 hours using the torch + kernels backend). The `batch_size` in the
config is per-device, so you should be able to run this on any number of GPUs
(although hyperparameters like learning rate are often sensitive to global batch
size, so keep in mind).

## Phonon fine-tuning (PFT)


First sync extra dependencies with

```bash
uv sync --extra pft
```

### Phonon calculations

We provide pretrained model weights for the co-trained (better alignment with
MPtrj) and non co-trained models in `models/nequix-mp-1-pft.nqx` and
`nequix-mp-1-pft-nocotrain.nqx` respectively. See [nequix-examples](https://github.com/teddykoker/nequix-examples) for
examples on how to use these models for phonon calculations with both finite
displacement, and analytical Hessians.


### Training

Data for the PBE MDR phonon database was originally downloaded and preprocessed with:

```bash
bash data/download_pbe_mdr.sh
uv run data/split_pbe_mdr.py
uv run scripts/preprocess_data_phonopy.py data/pbe-mdr/train data/pbe-mdr/train-aselmdb
uv run scripts/preprocess_data_phonopy.py data/pbe-mdr/val data/pbe-mdr/val-aselmdb
```

However we provide preprocessed data which can be downloaded with

```bash
bash data/download_pbe_mdr_preprocessed.sh
```

To run PFT without co-training run:

```bash
uv run nequix/pft/train.py configs/nequix-mp-1-pft-no-cotrain.yml
```

To run PFT *with* co-training run (note this requires `mptrj-aselmdb` preprocessed): 

```bash
uv run nequix/pft/train.py configs/nequix-mp-1-pft.yml
```

To run PFT on the OAM base model, follow the data download instructions below and then run:

```bash
uv run nequix/pft/train.py configs/nequix-oam-1-pft.yml
```

Both PFT training runs take about 140 hours on a single A100.

## Training OMat/OAM base models

To reproduce our training runs for the OMat and OAM base models run the following. First download OMat and sAlex data:


```bash
./data/download_omat.sh <path to storage location>
```

Then symlink to `./data`

```bash
ln -s <path to storage location>/omat ./data/omat
ln -s <path to storage location>/salex ./data/salex
ln -s <path to storage location>/mptrj-aselmdb ./data/mptrj-aselmdb
```

To train the OMat model, run:
```bash
uv run torchrun --nproc_per_node=4 nequix/torch/train.py configs/nequix-omat-1.yml
```

This takes roughly 60 hours on a 4 x A100 node. To fine-tune the OAM model, copy
the OMat model to `models/nequix-omat-1.pt` and run
```bash
uv run torchrun --nproc_per_node=4 nequix/torch/train.py configs/nequix-oam-1.yml
```
This takes roughly 10 hours on a 4 x A100 node.


## Citation

```bibtex
@article{koker2026pft,
  title={{PFT}: Phonon Fine-tuning for Machine Learned Interatomic Potentials},
  author={Koker, Teddy and Gangan, Abhijeet and Kotak, Mit and Marian, Jaime and Smidt, Tess},
  journal={arXiv preprint arXiv:2601.07742},
  year={2026}
}

@article{koker2025training,
  title={Training a foundation model for materials on a budget},
  author={Koker, Teddy and Kotak, Mit and Smidt, Tess},
  journal={arXiv preprint arXiv:2508.16067},
  year={2025}
}
```
