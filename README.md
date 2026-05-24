# Crowd Flow Prediction

This repository contains a PyTorch/PyTorch Lightning project for crowd counting through density-map estimation on standard crowd-counting datasets. The code trains encoder-decoder models that predict a single-channel density map from an RGB image; the predicted crowd count is obtained by summing the density map.


## How The Code Works

1. `CrowdCountingDataset` loads crowd images and point annotations from ShanghaiTech or UCF-QNRF-style folders.
2. Head coordinates are resized to the configured density-map resolution.
3. `create_density_map` converts points into a density map by placing impulses at head locations and applying a Gaussian filter.
4. `CrowdCountingDataModule` creates train/validation/test dataloaders, including optional cross-dataset test loaders.
5. `LitDensityEstimator` wraps any registered density-estimation model in the same PyTorch Lightning training/evaluation logic.
6. `benchmark.py` defines the shared dataset, optimization, trainer, logging, ablation, and model-grid configuration used for comparative experiments.
7. `train.py` is a CLI entrypoint for the unified benchmark grid.
8. `eval_baseline.py` evaluates mean-count and zero-density baselines.

## Key Features

- The main task is crowd counting via density-map estimation.
- The dataset interface targets ShanghaiTech Part A/B and UCF-QNRF point annotation datasets.
- The primary model families are pretrained ResNet50 and VGG19-BN encoders adapted into U-Net-like encoder-decoder networks.
- The benchmark can split depth/receptive-field, output-resolution, and skip-placement ablations instead of changing them together.
- Baselines include a mean-count density predictor and an all-zero density predictor.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── scripts/                       # Reproducibility scripts
│   ├── explore_ablations.sh
│   ├── explore_augmentation_diagnosis.sh
│   ├── explore_mixed_augmentation.sh
│   ├── explore_transfer_learning.sh
│   ├── prepare_datasets.sh
│   ├── reproduce_baselines.sh
│   ├── reproduce_table_1.sh
│   └── run_metric_tests.sh
├── data/                          # expected local dataset roots, not committed
│   ├── ShanghaiTech/
│   └── UCF-QNRF/
├── notebooks/                     # Jupyter notebooks for Exploratory Data Analysis (EDA)
│   ├── ShanghaiTech_EDA.ipynb
│   └── UCF_QNRF_EDA.ipynb
├── src/
│   ├── analyze_augmentation_distribution.py # script to diagnose dataset shifts
│   ├── benchmark.py               # unified benchmark harness and experiment grid
│   ├── data_loader.py             # multi-dataset crowd-counting data module
│   ├── eval_baseline.py           # mean-count and zero-density baseline evaluation
│   ├── metrics.py                 # pixelwise and count metrics
│   ├── train.py                   # benchmark CLI entrypoint
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
    ├── test_data_loader_multi_dataset.py
    └── test_evaluator.py          # synthetic evaluator and density-resize tests
```

## Reproducibility

To ensure strict adherence to modern public crowd-counting reproducibility standards (e.g., CLIP-EBC, STEERER), this repository guarantees exact reproducibility of the presented results, configurations, and evaluation environments.

- **Fixed Seeds:** All experiments utilize a fixed global random seed (`42`) set via `pytorch_lightning.seed_everything` inside `src/benchmark.py`. This ensures reproducible data splitting, cropping, and model initialization.
- **Preprocessing:** Density map generation is handled entirely on-the-fly inside `src/data_loader.py` using geometry-adaptive Gaussian kernels to map ground-truth coordinate points. Thus, offline preprocessing is not required.
- **Figures:** Density map visualizations are logged automatically to Weights & Biases during the first batch of every validation epoch.
- **Logs & Checkpoints:** When training via the reproducibility scripts, all model checkpoints are automatically exported to `models/checkpoints/` and detailed progress is tracked on Weights & Biases for easy artifact validation. 
- **Exact Scripts:** We provide the exact execution commands as explicit shell scripts inside the `scripts/` directory:

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download datasets locally and place them under `data/`. Supported dataset names are `sha`, `shb`, and `qnrf`.

### Dataset Downloads

These datasets are not committed to the repository. Download them from the dataset owners or mirrors that preserve the original archives:

| Dataset | Root folder | Approx. archive size | Download page |
| --- | --- | ---: | --- |
| ShanghaiTech Part A/B | `data/ShanghaiTech/` | varies by mirror | <https://www.kaggle.com/datasets/tthien/shanghaitech> |
| UCF-QNRF | `data/UCF-QNRF/` | 4.24 GB | <https://www.crcv.ucf.edu/data/ucf-qnrf/> |

Many official links are hosted on Google Drive, OneDrive, Dropbox, Baidu, or web portals that may require browser confirmation, login, quota availability, or accepting terms. In those cases, download the archives manually in a browser, then extract them into the folders above. Keep the original archives outside git; `data/` is ignored by `.gitignore`.

UCF-QNRF currently has a direct ZIP on the UCF site:

```bash
mkdir -p data
curl --insecure --http1.1 -L --fail --continue-at - --retry 5 --retry-delay 5 \
  -o data/UCF-QNRF_ECCV18.zip \
  https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip
unzip -q data/UCF-QNRF_ECCV18.zip -d data
mv data/UCF-QNRF_ECCV18 data/UCF-QNRF
rm data/UCF-QNRF_ECCV18.zip
```



After extraction, verify the loader can see a split:

```bash
python - <<'PY'
from src.data_loader import CrowdCountingDataset

checks = [
    ("./data/UCF-QNRF", "qnrf", "train"),
]

for root, name, split in checks:
    try:
        ds = CrowdCountingDataset(root=root, dataset_name=name, split=split)
        print(f"{name}:{split}: {len(ds)} images")
    except FileNotFoundError as exc:
        print(f"{name}:{split}: not found ({exc})")
PY
```

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

For UCF-QNRF-style datasets, use this shape when possible:

```text
data/UCF-QNRF/
├── train/
│   ├── images/
│   └── ground_truth/
└── test/
    ├── images/
    └── ground_truth/
```

The loader also accepts common real-world variants:

```text
data/UCF-QNRF/
├── Train/
│   ├── img_0001.jpg
│   └── img_0001_ann.mat
└── Test/
    ├── img_0001.jpg
    └── img_0001_ann.mat
```

The loader accepts common annotation names and formats such as `GT_<image>.mat`, `<image>.mat`, `<image>_ann.mat`, `<image>.txt`, `<image>.csv`, and `<image>.json`. It looks for common point keys including `image_info`, `annPoints`, `points`, and `locations`.

## Complete Experiment Overview

We have encapsulated all evaluation, baseline testing, ablation studies, and augmentation explorations into a set of sequential bash scripts. Running these scripts in order provides a complete and reproducible overview of all findings, including those detailed in the paper and additional generalized explorations.

**1. Data Preparation**
Download and extract the required datasets (e.g., UCF-QNRF). Follow manual download instructions for datasets behind web portals (e.g. ShanghaiTech) as described in the Setup section.
```bash
./scripts/prepare_datasets.sh
```

**2. Validate Metrics**
Run the unit test suite to verify the mathematical correctness of quantitative metrics (Empty-Region FP Mass, Dense Region MSE, GAME).
```bash
./scripts/run_metric_tests.sh
```

**3. Evaluate Baselines**
Reproduce the "Zeros" and "Mean" density baselines.
```bash
./scripts/reproduce_baselines.sh
```

**4. Reproduce Main Paper Results (Table 1)**
Train and evaluate the core VGG19 and ResNet50 models across varying depths on ShanghaiTech Parts A and B. This command will log the resulting density map figures to Weights & Biases.
```bash
./scripts/reproduce_table_1.sh
```

**5. Explore Architecture Ablations**
Isolate and examine the specific impacts of receptive field, output resolution, and skip-connection placements using controlled ablations.
```bash
./scripts/explore_ablations.sh
```

**6. Explore Transfer Learning**
Train a baseline model on internet scenes (ShanghaiTech Part A) and evaluate its zero-shot generalization capabilities on high-resolution, unconstrained crowds (UCF-QNRF).
```bash
./scripts/explore_transfer_learning.sh
```

**7. Diagnose Crop Augmentation**
Analyze how naive random cropping causes severe distribution shifts (e.g. creating excessive empty patches), which leads to generalization failure. This script outputs distribution statistics and plots for UCF-QNRF.
```bash
./scripts/explore_augmentation_diagnosis.sh
```

**8. Train with Advanced Mixed Augmentation**
Train on UCF-QNRF using the disciplined mixed augmentation strategy (combining full images with carefully balanced crops) designed to overcome the distribution shifts diagnosed in the previous step.
```bash
./scripts/explore_mixed_augmentation.sh
```

*Note: All model variants in this pipeline share the same preprocessing, optimizer, scheduler, callbacks, output resizing, metrics, and checkpoint policy. The training scripts automatically log to Weights & Biases and write checkpoints under `models/checkpoints/`.*