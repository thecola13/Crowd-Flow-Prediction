#!/usr/bin/env bash
set -e

# This script downloads and extracts UCF-QNRF.
# For ShanghaiTech, NWPU, and JHU, you must manually download them from the official sources
# as they require web portal authentication or Captchas.
# Check README.md for the official links.

echo "Creating data directories..."
mkdir -p data

echo "Downloading UCF-QNRF..."
curl --http1.1 -L --fail --continue-at - --retry 5 --retry-delay 5 \
  -o data/UCF-QNRF_ECCV18.zip \
  https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip

echo "Extracting UCF-QNRF..."
unzip -q data/UCF-QNRF_ECCV18.zip -d data
mv data/UCF-QNRF_ECCV18 data/UCF-QNRF

echo "UCF-QNRF preparation complete."
echo "Please manually download ShanghaiTech, NWPU, and JHU datasets as per README.md instructions."
