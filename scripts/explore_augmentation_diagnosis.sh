#!/usr/bin/env bash
set -e

echo "Running Crop Augmentation Diagnosis on UCF-QNRF"
python -m src.analyze_augmentation_distribution \
  --data-folder ./data/UCF-QNRF \
  --dataset qnrf \
  --split train \
  --max-images 150 \
  --crops-per-image 2 \
  --crop-size 256x256 \
  --output-json ./outputs/augmentation_distribution_ucf_qnrf_train.json \
  --plot-dir ./outputs/augmentation_distribution_ucf_qnrf_train_plots

echo "Running Crop Augmentation Diagnosis on ShanghaiTech Part A"
python -m src.analyze_augmentation_distribution \
  --data-folder ./data/ShanghaiTech \
  --dataset sha \
  --split train \
  --max-images 150 \
  --crops-per-image 2 \
  --crop-size 256x256 \
  --output-json ./outputs/augmentation_distribution_shanghaitech_a_train.json \
  --plot-dir ./outputs/augmentation_distribution_shanghaitech_a_train_plots

echo "Diagnosis complete. Check ./outputs/ for the JSON reports and plots."
