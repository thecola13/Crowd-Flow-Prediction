#!/usr/bin/env bash
set -e

# Reproduce the full benchmark grid presented in Table 1
# This evaluates VGG19 and ResNet50 at Depths 2, 3, and 4 on ShanghaiTech Parts A and B.

echo "Reproducing Table 1: Training and Evaluation Grid on ShanghaiTech"
python -m src.train \
    --architectures vgg19_ae,resnet50_ae \
    --depths 2,3,4 \
    --splits A,B \
    --data-folder ./data/ShanghaiTech \
    --dataset sha

echo "Reproducing Table 1: Training and Evaluation Grid on UCF-QNRF"
python -m src.train \
    --architectures vgg19_ae,resnet50_ae \
    --depths 2,3,4 \
    --splits qnrf \
    --data-folder ./data/UCF-QNRF \
    --dataset qnrf

echo "Table 1 Grid reproduction complete. Check weights and biases logs or test stdout for MAE/RMSE."
