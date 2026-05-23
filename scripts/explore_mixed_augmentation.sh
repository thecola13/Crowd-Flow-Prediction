#!/usr/bin/env bash
set -e

echo "Training on UCF-QNRF with Advanced Mixed Augmentation"

# Added an optimized learning rate of 2e-4, suitable for learning from heavily augmented data
python -m src.train \
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

echo "Mixed augmentation training complete."
