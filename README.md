# Crowd Flow Prediction

This repository contains a PyTorch/PyTorch Lightning project for crowd counting through density-map estimation on the ShanghaiTech crowd counting dataset. The code trains encoder-decoder models that predict a single-channel density map from an RGB image; the predicted crowd count is obtained by summing the density map.

The accompanying report in `latex/main.tex` studies how receptive field size affects density-map quality and count accuracy. It compares VGG19-BN and ResNet50 encoder variants at different depths on ShanghaiTech Part A and Part B.

## How The Code Works

1. `ShanghaiTechDataset` loads ShanghaiTech images and `.mat` head annotations from `data/ShanghaiTech/<part>/<split>/`.
2. Head coordinates are resized to the configured density-map resolution.
3. `create_density_map` converts points into a density map by placing impulses at head locations and applying a Gaussian filter.
4. `ShanghaiTechDataModule` creates train/validation/test dataloaders.
5. `LitDensityEstimator` wraps any registered density-estimation model in the same PyTorch Lightning training/evaluation logic.
6. `benchmark.py` defines the shared dataset, optimization, trainer, logging, and model-grid configuration used for comparative experiments.
7. `train.py` is a CLI entrypoint for the unified benchmark grid.
8. `eval_baseline.py` evaluates mean-count and zero-density baselines.

## Alignment With The Report

The implementation is broadly aligned with the report:

- The main task is crowd counting via density-map estimation.
- The dataset interface targets ShanghaiTech Part A and Part B.
- The primary model families are pretrained ResNet50 and VGG19-BN encoders adapted into U-Net-like encoder-decoder networks.
- The training grid varies model depth to study receptive field effects.
- The code computes count MAE/RMSE and pixelwise MAE/RMSE; the report currently presents only count metrics while pixel-level evaluation is being revalidated.
- Baselines include a mean-count density predictor and an all-zero density predictor.

Important differences and caveats:

- The report describes geometry-adaptive Gaussian kernels, while `src/utils.py` currently uses a fixed Gaussian `sigma`.
- The implemented ResNet/VGG models output half-resolution density maps when the input is `384x384` and the density target is `192x192`.
- The README and requirements were previously minimal; setup and run instructions below reflect the code as it exists now.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── ShanghaiTech/              # expected local dataset location, not committed
├── latex/
│   ├── main.tex                   # project report
│   ├── references.bib
│   └── images/                    # architecture and result figures used by the report
├── src/
│   ├── data_loader.py             # ShanghaiTech dataset and Lightning data module
│   ├── benchmark.py               # unified benchmark harness and experiment grid
│   ├── eval_baseline.py           # mean-count and zero-density baseline evaluation
│   ├── metrics.py                 # pixelwise and count metrics
│   ├── train.py                   # experiment runner
│   ├── train_lightning.py         # LightningModule for density estimation
│   ├── utils.py                   # density-map creation, plotting, device, receptive field utility
│   └── models/
│       ├── __init__.py            # model factory
│       ├── resnet50.py            # ResNet50 encoder-decoder model
│       ├── unet.py                # generic U-Net model
│       ├── unet_comp.py           # reusable U-Net blocks
│       └── vgg19bn.py             # VGG19-BN encoder-decoder model
└── tests/
    ├── test_benchmark_config.py   # benchmark grid/configuration tests
    └── test_evaluator.py          # synthetic evaluator and density-resize tests
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the ShanghaiTech dataset locally and place it under:

```text
data/ShanghaiTech/
├── part_A/
│   ├── train_data/
│   │   ├── images/
│   │   └── ground_truth/
│   └── test_data/
│       ├── images/
│       └── ground_truth/
└── part_B/
    ├── train_data/
    │   ├── images/
    │   └── ground_truth/
    └── test_data/
        ├── images/
        └── ground_truth/
```

## Running Experiments

Evaluate baselines:

```bash
python -m src.eval_baseline
```

Run unit tests:

```bash
python -m unittest discover -s tests
```

Train the full experiment grid:

```bash
python -m src.train
```

Run a smaller controlled benchmark:

```bash
python -m src.train --architectures resnet50_ae,vgg19_ae --depths 4 --splits A --no-wandb
```

All model variants in this entrypoint share the same preprocessing, optimizer, scheduler, callbacks, output resizing, metrics, and checkpoint policy. The architecture aliases currently supported by the repo are `resnet50_ae`, `vgg19_ae`, and `unet`.

The training script logs to Weights & Biases and writes checkpoints under `models/checkpoints/`.

## Notes

- Default image size is `384x384`; default density-map size in training is `192x192`.
- Counts are computed as the spatial sum of the density map.
- The current train/validation split is random within `train_data`, controlled by a fixed seed.
- Generated outputs, checkpoints, and local datasets should stay out of version control.
