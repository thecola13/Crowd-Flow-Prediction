#!/usr/bin/env bash
set -e

# Runs the unit test suite, including tests for the quantitative metrics
# (Empty-Region FP Mass, Dense Region MSE, GAME)

# Determine the python executable to use
if [ -f "./.venv/bin/python" ]; then
    PYTHON_EXE="./.venv/bin/python"
else
    PYTHON_EXE="python"
fi

echo "Running unit tests and metric verifications..."
$PYTHON_EXE -m unittest discover -s tests

echo "Metric tests completed successfully."
