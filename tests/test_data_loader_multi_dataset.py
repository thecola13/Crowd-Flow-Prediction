import tempfile
import unittest
import json
from pathlib import Path

import scipy.io
import torch
from PIL import Image

from src.data_loader import CrowdCountingDataset


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

    def test_nwpu_root_layout_with_split_manifest_and_mats_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NWPU-Crowd"
            write_image(root / "images" / "0001.jpg")
            write_image(root / "images" / "0002.jpg")
            (root / "train.txt").write_text("0002 0 1\n", encoding="utf-8")
            mats_dir = root / "mats"
            mats_dir.mkdir(parents=True, exist_ok=True)
            scipy.io.savemat(mats_dir / "0001.mat", {"annPoints": [[1.0, 1.0]]})
            scipy.io.savemat(
                mats_dir / "0002.mat",
                {"annPoints": [[2.0, 2.0], [5.0, 5.0]]},
            )

            dataset = CrowdCountingDataset(
                root=str(root),
                dataset_name="nwpu",
                split="train",
                input_size=(8, 8),
                density_map_size=(8, 8),
                sigma=0,
            )
            _, density = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.image_files[0].name, "0002.jpg")
            self.assertAlmostEqual(density.sum().item(), 2.0)

    def test_nwpu_layout_with_txt_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "NWPU"
            write_image(root / "test" / "images" / "001.jpg")
            gt_dir = root / "test" / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            (gt_dir / "001.txt").write_text("2 2\n5 5\n", encoding="utf-8")

            dataset = CrowdCountingDataset(
                root=str(root),
                dataset_name="nwpu",
                split="test",
                input_size=(8, 8),
                density_map_size=(4, 4),
                sigma=0,
            )
            _, density = dataset[0]

            self.assertEqual(tuple(density.shape), (1, 4, 4))
            self.assertTrue(torch.isclose(density.sum(), torch.tensor(2.0)))

    def test_jhu_layout_with_json_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "JHU"
            write_image(root / "train" / "images" / "scene.png")
            gt_dir = root / "train" / "annotations"
            gt_dir.mkdir(parents=True, exist_ok=True)
            (gt_dir / "scene.json").write_text(
                json.dumps({"points": [[1, 1], [3, 3], [6, 6]]}),
                encoding="utf-8",
            )

            dataset = CrowdCountingDataset(
                root=str(root),
                dataset_name="jhu",
                split="train",
                input_size=(8, 8),
                density_map_size=(8, 8),
                sigma=0,
            )
            _, density = dataset[0]

            self.assertAlmostEqual(density.sum().item(), 3.0)

    def test_jhu_text_rows_keep_first_two_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "JHU"
            write_image(root / "val" / "images" / "0001.jpg")
            gt_dir = root / "val" / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            (gt_dir / "0001.txt").write_text(
                "1 1 8 8 0 0\n3 3 10 10 1 0\n",
                encoding="utf-8",
            )

            dataset = CrowdCountingDataset(
                root=str(root),
                dataset_name="jhu",
                split="val",
                input_size=(8, 8),
                density_map_size=(8, 8),
                sigma=0,
            )
            _, density = dataset[0]

            self.assertAlmostEqual(density.sum().item(), 2.0)


if __name__ == "__main__":
    unittest.main()
