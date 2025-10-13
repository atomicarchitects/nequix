<h1 align='center'>Nequix</h1>

See more information in our [preprint](https://arxiv.org/abs/2508.16067).

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

Then start the training run. The first time this is run it will preprocess the data into HDF5 files:

```bash
nequix_train configs/nequix-mp-1.yml
```

This will take less than 125 hours on a single 4 x A100 node (<25 hours using the torch + kernels backend). The `batch_size` in the
config is per-device, so you should be able to run this on any number of GPUs
(although hyperparameters like learning rate are often sensitive to global batch
size, so keep in mind).


## Citation

```bibtex
@article{koker2025training,
  title={Training a foundation model for materials on a budget},
  author={Koker, Teddy and Kotak, Mit and Smidt, Tess},
  journal={arXiv preprint arXiv:2508.16067},
  year={2025}
}
```
