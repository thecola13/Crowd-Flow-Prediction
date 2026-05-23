import tempfile
import unittest
import json
from pathlib import Path

import scipy.io
import torch
from PIL import Image

from src.data_loader import (
    AugmentedCrowdCountingDataset,
    CropAugmentationConfig,
    CrowdCountingDataModule,
    CrowdCountingDataset,
)


def write_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(255, 255, 255)).save(path)


class MultiDatasetLoaderTests(unittest.TestCase):
    def test_qnrf_layout_with_annpoints_mat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "UCF-QNRF"
            write_image(root / "Train" / "img_0001.jpg")
            scipy.io.savemat(
                root / "Train" / "img_0001_ann.mat",
                {"annPoints": [[1.0, 1.0], [6.0, 6.0]]},
            )

            dataset = CrowdCountingDataset(
                root=str(root),
                dataset_name="qnrf",
                split="train",
                input_size=(8, 8),
                density_map_size=(8, 8),
                sigma=0,
            )
            _, density = dataset[0]

            self.assertEqual(tuple(density.shape), (1, 8, 8))
            self.assertAlmostEqual(density.sum().item(), 2.0)



    def test_balanced_crop_sampler_preserves_crop_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "UCF-QNRF"
            write_image(root / "Train" / "img_0001.jpg")
            scipy.io.savemat(
                root / "Train" / "img_0001_ann.mat",
                {"annPoints": [[1.0, 1.0], [6.0, 6.0]]},
            )
            dataset = CrowdCountingDataset(
                root=str(root),
                dataset_name="qnrf",
                split="train",
                input_size=(8, 8),
                density_map_size=(8, 8),
                sigma=0,
            )
            augmented = AugmentedCrowdCountingDataset(
                dataset,
                CropAugmentationConfig(
                    enabled=True,
                    crop_size=(8, 8),
                    full_image_probability=0.0,
                    scale_jitter=(1.0, 1.0),
                    horizontal_flip_probability=0.0,
                    photometric_jitter=0.0,
                ),
                seed=1,
            )

            _, density = augmented[0]

            self.assertAlmostEqual(density.sum().item(), 2.0)

    def test_data_module_uses_augmented_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "UCF-QNRF"
            for idx in range(3):
                write_image(root / "Train" / f"img_000{idx}.jpg")
                scipy.io.savemat(
                    root / "Train" / f"img_000{idx}_ann.mat",
                    {"annPoints": [[1.0, 1.0], [6.0, 6.0]]},
                )
                write_image(root / "Test" / f"img_000{idx}.jpg")
                scipy.io.savemat(
                    root / "Test" / f"img_000{idx}_ann.mat",
                    {"annPoints": [[1.0, 1.0]]},
                )
            data_module = CrowdCountingDataModule(
                data_folder=str(root),
                dataset_name="qnrf",
                validation_split=1 / 3,
                batch_size=1,
                num_workers=0,
                input_size=(8, 8),
                density_map_size=(8, 8),
                sigma=0,
                crop_augmentation=CropAugmentationConfig(
                    enabled=True,
                    crops_per_image=2,
                    full_image_probability=0.0,
                    scale_jitter=(1.0, 1.0),
                    horizontal_flip_probability=0.0,
                    photometric_jitter=0.0,
                ),
            )

            data_module.setup()

            self.assertIsInstance(data_module.train_dataset, AugmentedCrowdCountingDataset)
            self.assertEqual(len(data_module.train_dataset), 4)
            self.assertEqual(len(data_module.val_dataset), 1)


if __name__ == "__main__":
    unittest.main()
