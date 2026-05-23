#!/usr/bin/env bash
set -e

# Runs the unit test suite, including tests for the quantitative metrics
# (Empty-Region FP Mass, Dense Region MSE, GAME)

echo "Running unit tests and metric verifications..."
python -m unittest discover -s tests

echo "Metric tests completed successfully."
