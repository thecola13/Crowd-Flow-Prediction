#!/usr/bin/env bash
set -e

# Determine the python executable to use
if [ -f "./.venv/bin/python" ]; then
    PYTHON_EXE="./.venv/bin/python"
else
    PYTHON_EXE="python"
fi

echo "Training on UCF-QNRF with Advanced Mixed Augmentation"
# Added an optimized learning rate of 2e-4, suitable for learning from heavily augmented data
$PYTHON_EXE -m src.train \
  --dataset qnrf \
  --data-folder ./data/UCF-QNRF \
  --architectures vgg19_ae \
  --depths 4 \
  --splits qnrf_aug \
  --use-crop-augmentation \
  --crops-per-image 2 \
  --full-image-probability 0.5 \
  --crop-size 256x256 \
  --scale-jitter 0.75,1.25 \
  --horizontal-flip-probability 0.5 \
  --photometric-jitter 0.15 \
  --batch-size 8 \
  --lr 2e-4 \
  --max-epochs 250

echo "Training on ShanghaiTech Part A with Advanced Mixed Augmentation"
$PYTHON_EXE -m src.train \
  --dataset sha \
  --data-folder ./data/ShanghaiTech \
  --architectures vgg19_ae \
  --depths 4 \
  --splits sha_aug \
  --use-crop-augmentation \
  --crops-per-image 2 \
  --full-image-probability 0.5 \
  --crop-size 256x256 \
  --scale-jitter 0.75,1.25 \
  --horizontal-flip-probability 0.5 \
  --photometric-jitter 0.15 \
  --batch-size 8 \
  --lr 2e-4 \
  --max-epochs 250

echo "Mixed augmentation training complete."
