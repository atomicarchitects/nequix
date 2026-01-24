<h1 align='center'>PFT</h1>

Source code for [Phonon fine-tuning (PFT)](https://arxiv.org/abs/2601.07742).

## Usage

First sync dependencies with

```bash
uv sync
```

### Phonon Calculations

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


