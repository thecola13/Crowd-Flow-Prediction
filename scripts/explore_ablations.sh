#!/usr/bin/env bash
set -e

echo "Running controlled ablations separately..."

echo "1. Receptive Field Ablation (ResNet50, Depths 2-4)"
python -m src.train \
    --ablation receptive_field \
    --architectures resnet50_ae \
    --depths 2,3,4 \
    --output-reductions 2 \
    --splits A

echo "2. Output Resolution Ablation (VGG19, Reductions 1,2,4)"
python -m src.train \
    --ablation output_resolution \
    --architectures vgg19_ae \
    --depths 4 \
    --output-reductions 1,2,4 \
    --splits A

echo "3. Skip Placement Ablation (UNet, Before/After Pool)"
python -m src.train \
    --ablation skip_placement \
    --architectures unet \
    --depths 4 \
    --output-reductions 2 \
    --splits A

echo "Ablations complete."
