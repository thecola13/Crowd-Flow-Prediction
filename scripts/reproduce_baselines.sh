#!/usr/bin/env bash
set -e

# Reproduce the Mean and Zeros baseline evaluation metrics for Table 1

# Determine the python executable to use
if [ -f "./.venv/bin/python" ]; then
    PYTHON_EXE="./.venv/bin/python"
else
    PYTHON_EXE="python"
fi

echo "Reproducing Mean and Zeros Baselines"
$PYTHON_EXE -m src.eval_baseline

echo "Baseline reproduction complete."
