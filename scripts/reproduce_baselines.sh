#!/usr/bin/env bash
set -e

# Reproduce the Mean and Zeros baseline evaluation metrics for Table 1

echo "Reproducing Mean and Zeros Baselines"
python -m src.eval_baseline

echo "Baseline reproduction complete."
