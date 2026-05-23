# Crowd Flow Prediction

This repository contains a PyTorch/PyTorch Lightning project for crowd counting through density-map estimation on standard crowd-counting datasets. The code trains encoder-decoder models that predict a single-channel density map from an RGB image; the predicted crowd count is obtained by summing the density map.

The accompanying report in `latex/main.tex` studies how receptive field size affects density-map quality and count accuracy. It compares VGG19-BN and ResNet50 encoder variants at different depths on ShanghaiTech Part A and Part B.

## How The Code Works

1. `CrowdCountingDataset` loads crowd images and point annotations from ShanghaiTech, UCF-QNRF, NWPU, or JHU-style folders.
2. Head coordinates are resized to the configured density-map resolution.
3. `create_density_map` converts points into a density map by placing impulses at head locations and applying a Gaussian filter.
4. `CrowdCountingDataModule` creates train/validation/test dataloaders, including optional cross-dataset test loaders.
5. `LitDensityEstimator` wraps any registered density-estimation model in the same PyTorch Lightning training/evaluation logic.
6. `benchmark.py` defines the shared dataset, optimization, trainer, logging, ablation, and model-grid configuration used for comparative experiments.
7. `train.py` is a CLI entrypoint for the unified benchmark grid.
8. `eval_baseline.py` evaluates mean-count and zero-density baselines.

## Alignment With The Report

The implementation is broadly aligned with the report:

- The main task is crowd counting via density-map estimation.
- The dataset interface targets ShanghaiTech Part A/B, UCF-QNRF, NWPU, and JHU-style point annotation datasets.
- The primary model families are pretrained ResNet50 and VGG19-BN encoders adapted into U-Net-like encoder-decoder networks.
- The benchmark can now split depth/receptive-field, output-resolution, and skip-placement ablations instead of changing them together.
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
├── data/                          # expected local dataset roots, not committed
│   ├── ShanghaiTech/
│   ├── UCF-QNRF/
│   ├── NWPU/
│   └── JHU/
├── latex/
│   ├── main.tex                   # project report
│   ├── references.bib
│   └── images/                    # architecture and result figures used by the report
├── src/
│   ├── data_loader.py             # multi-dataset crowd-counting data module
│   ├── benchmark.py               # unified benchmark harness and experiment grid
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

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download datasets locally and place them under `data/`. Supported dataset names are `sha`, `shb`, `qnrf`, `nwpu`, and `jhu`.

### Dataset Downloads

These datasets are not committed to the repository. Download them from the dataset owners or mirrors that preserve the original archives:

| Dataset | Root folder | Approx. archive size | Download page |
| --- | --- | ---: | --- |
| ShanghaiTech Part A/B | `data/ShanghaiTech/` | varies by mirror | <https://www.kaggle.com/datasets/tthien/shanghaitech> |
| UCF-QNRF | `data/UCF-QNRF/` | 4.24 GB | <https://www.crcv.ucf.edu/data/ucf-qnrf/> |
| JHU-CROWD++ | `data/JHU/` | 2.87 GB | <http://www.crowd-counting.com/> |
| NWPU-Crowd | `data/NWPU/` | component archives; full image set is large | <https://gjy3035.github.io/NWPU-Crowd-Sample-Code/> |

If you need to stay under roughly 10 GB, download UCF-QNRF and JHU-CROWD++ first. Together they are about 7.11 GB before extraction. For NWPU-Crowd, download only the split files plus annotation archives (`mats.zip` or `jsons.zip`) and a small image subset from the official component links, or use the official sample download, then place those files under `data/NWPU/`.

Many official links are hosted on Google Drive, OneDrive, Dropbox, Baidu, or web portals that may require browser confirmation, login, quota availability, or accepting terms. In those cases, download the archives manually in a browser, then extract them into the folders above. Keep the original archives outside git; `data/` is ignored by `.gitignore`.

UCF-QNRF currently has a direct ZIP on the UCF site:

```bash
mkdir -p data
curl --http1.1 -L --fail --continue-at - --retry 5 --retry-delay 5 \
  -o data/UCF-QNRF_ECCV18.zip \
  https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip
unzip -q data/UCF-QNRF_ECCV18.zip -d data
mv data/UCF-QNRF_ECCV18 data/UCF-QNRF
```

JHU-CROWD++ exposes Dropbox, Google Drive, and OneDrive links from the official page. NWPU-Crowd exposes OneDrive, BaiduNetdisk, and CrowdBenchmark folders from the official page. Prefer downloading those in a browser if the command line receives an HTML/JSON confirmation page instead of the archive.

After extraction, verify the loader can see a split:

```bash
python - <<'PY'
from src.data_loader import CrowdCountingDataset

checks = [
    ("./data/UCF-QNRF", "qnrf", "train"),
    ("./data/JHU", "jhu", "train"),
    ("./data/NWPU", "nwpu", "train"),
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

For UCF-QNRF/NWPU/JHU-style datasets, use this shape when possible:

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

data/NWPU/
├── images/
├── mats/
├── jsons/
├── train.txt
├── val.txt
└── test.txt
```

The loader accepts common annotation names and formats such as `GT_<image>.mat`, `<image>.mat`, `<image>_ann.mat`, `<image>.txt`, `<image>.csv`, and `<image>.json`. It looks for common point keys including `image_info`, `annPoints`, `points`, and `locations`. For multi-column text annotations such as JHU-CROWD++, the first two columns are interpreted as `(x, y)` point coordinates.

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

Run the three cleaner ablations separately:

```bash
python -m src.train --ablation receptive_field --architectures resnet50_ae --depths 2,3,4 --output-reductions 2 --splits A
python -m src.train --ablation output_resolution --architectures vgg19_ae --depths 4 --output-reductions 1,2,4 --splits A
python -m src.train --ablation skip_placement --architectures unet --depths 4 --output-reductions 2 --splits A
```

Run on another dataset:

```bash
python -m src.train --dataset qnrf --data-folder ./data/UCF-QNRF --splits qnrf --architectures vgg19_ae --depths 4 --output-reductions 2
```

Run a transfer-style evaluation, training on ShanghaiTech Part A and testing on UCF-QNRF:

```bash
python -m src.train --dataset sha --data-folder ./data/ShanghaiTech --eval-dataset qnrf --eval-data-folder ./data/UCF-QNRF --splits sha_to_qnrf --architectures vgg19_ae --depths 4
```

All model variants in this entrypoint share the same preprocessing, optimizer, scheduler, callbacks, output resizing, metrics, and checkpoint policy. Output reduction is an explicit model hyperparameter. The architecture aliases currently supported by the repo are `resnet50_ae`, `vgg19_ae`, and `unet`.

The training script logs to Weights & Biases and writes checkpoints under `models/checkpoints/`.

## Notes

- Default image size is `384x384`; density-map size is derived from `output_reduction` unless explicitly overridden, so the default reduction of `2` gives `192x192` targets.
- Counts are computed as the spatial sum of the density map.
- The current train/validation split is random within `train_data`, controlled by a fixed seed.
- Generated outputs, checkpoints, and local datasets should stay out of version control.
