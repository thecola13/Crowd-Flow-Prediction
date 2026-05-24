#!/usr/bin/env bash
set -e

# Determine the python executable to use
if [ -f "./.venv/bin/python" ]; then
    PYTHON_EXE="./.venv/bin/python"
else
    PYTHON_EXE="python"
fi

echo "Running controlled ablations separately..."

echo "1. Receptive Field Ablation (ResNet50, Depths 2-4)"
echo "=> Running on ShanghaiTech Part A"
$PYTHON_EXE -m src.train \
    --ablation receptive_field \
    --architectures resnet50_ae \
    --depths 2,3,4 \
    --output-reductions 2 \
    --splits A \
    --dataset sha \
    --data-folder ./data/ShanghaiTech

echo "=> Running on UCF-QNRF"
$PYTHON_EXE -m src.train \
    --ablation receptive_field \
    --architectures resnet50_ae \
    --depths 2,3,4 \
    --output-reductions 2 \
    --splits qnrf \
    --dataset qnrf \
    --data-folder ./data/UCF-QNRF

echo "2. Output Resolution Ablation (VGG19, Reductions 1,2,4)"
echo "=> Running on ShanghaiTech Part A"
$PYTHON_EXE -m src.train \
    --ablation output_resolution \
    --architectures vgg19_ae \
    --depths 4 \
    --output-reductions 1,2,4 \
    --splits A \
    --dataset sha \
    --data-folder ./data/ShanghaiTech

echo "=> Running on UCF-QNRF"
$PYTHON_EXE -m src.train \
    --ablation output_resolution \
    --architectures vgg19_ae \
    --depths 4 \
    --output-reductions 1,2,4 \
    --splits qnrf \
    --dataset qnrf \
    --data-folder ./data/UCF-QNRF

echo "3. Skip Placement Ablation (vgg19_ae, Before/After Pool)"
echo "=> Running on ShanghaiTech Part A"
$PYTHON_EXE -m src.train \
    --ablation skip_placement \
    --architectures vgg19_ae \
    --depths 4 \
    --output-reductions 2 \
    --splits A \
    --dataset sha \
    --data-folder ./data/ShanghaiTech

echo "=> Running on UCF-QNRF"
$PYTHON_EXE -m src.train \
    --ablation skip_placement \
    --architectures vgg19_ae \
    --depths 4 \
    --output-reductions 2 \
    --splits qnrf \
    --dataset qnrf \
    --data-folder ./data/UCF-QNRF

echo "Ablations complete."
