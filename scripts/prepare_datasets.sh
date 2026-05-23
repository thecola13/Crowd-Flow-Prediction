#!/usr/bin/env bash
set -e

# This script downloads and extracts UCF-QNRF.
# For ShanghaiTech, you must manually download it from the official sources
# as it requires web portal authentication or Captchas.
# Check README.md for the official links.

echo "Creating data directories..."
mkdir -p data

echo "Downloading UCF-QNRF..."
until curl --insecure --http1.1 -L --fail --continue-at - --retry 5 --retry-delay 5 \
  -o data/UCF-QNRF_ECCV18.zip \
  https://www.crcv.ucf.edu/data/ucf-qnrf/UCF-QNRF_ECCV18.zip; do
    echo "Connection dropped. Resuming download..."
    sleep 2
done

echo "Extracting UCF-QNRF..."
unzip -q data/UCF-QNRF_ECCV18.zip -d data
mv data/UCF-QNRF_ECCV18 data/UCF-QNRF
rm data/UCF-QNRF_ECCV18.zip

echo "UCF-QNRF preparation complete."
echo "Please manually download ShanghaiTech dataset as per README.md instructions."
