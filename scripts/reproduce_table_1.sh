#!/usr/bin/env bash
set -e

# Reproduce the full benchmark grid presented in Table 1
# This evaluates VGG19 and ResNet50 at Depths 2, 3, and 4 on ShanghaiTech Parts A and B.

echo "Reproducing Table 1: Training and Evaluation Grid"
python -m src.train \
    --architectures vgg19_ae,resnet50_ae \
    --depths 2,3,4 \
    --splits A,B \
    --data-folder ./data/ShanghaiTech

echo "Table 1 Grid reproduction complete. Check weights and biases logs or test stdout for MAE/RMSE."
