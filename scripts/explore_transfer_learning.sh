#!/usr/bin/env bash
set -e

echo "Exploring Transfer Learning Generalization"
echo "Training on ShanghaiTech Part A, Evaluating on UCF-QNRF"

# Lowering the learning rate slightly from the default (5e-4) to 1e-4 
# for more stable convergence when preparing for cross-dataset transfer evaluation.
python -m src.train \
    --dataset sha \
    --data-folder ./data/ShanghaiTech \
    --eval-dataset qnrf \
    --eval-data-folder ./data/UCF-QNRF \
    --splits sha_to_qnrf \
    --architectures vgg19_ae \
    --depths 4 \
    --lr 1e-4 \
    --max-epochs 200

echo "Transfer learning evaluation complete."
