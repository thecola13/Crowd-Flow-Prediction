import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    class _LightningDataModule:
        pass

    class _PL:
        LightningDataModule = _LightningDataModule

    pl = _PL()

from src.utils import create_density_map, get_device, show_samples_from_loaders


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
POINT_KEYS = (
    "annPoints",
    "points",
    "point",
    "locations",
    "image_info",
    "gt",
    "head",
    "heads",
)


def get_default_transform(input_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )


def _normalise_dataset_name(dataset_name: str) -> str:
    aliases = {
        "sha": "sha",
        "shanghai_a": "sha",
        "shanghaitech_a": "sha",
        "part_a": "sha",
        "shb": "shb",
        "shanghai_b": "shb",
        "shanghaitech_b": "shb",
        "part_b": "shb",
        "qnrf": "qnrf",
        "ucf_qnrf": "qnrf",
        "ucf-qnrf": "qnrf",
        "nwpu": "nwpu",
        "jhu": "jhu",
        "jhu_crowd": "jhu",
    }
    key = dataset_name.lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Supported values include sha, shb, qnrf, nwpu, and jhu."
        )
    return aliases[key]


def _candidate_split_names(dataset_name: str, split: str) -> list[str]:
    split_key = split.lower()
    if split_key in {"train", "train_data"}:
        base = ["train", "Train", "train_data", "Train_data"]
    elif split_key in {"val", "valid", "validation"}:
        base = ["val", "Val", "valid", "Valid", "validation", "Validation"]
    elif split_key in {"test", "test_data"}:
        base = ["test", "Test", "test_data", "Test_data"]
    else:
        base = [split, split.capitalize()]

    if dataset_name == "sha":
        return [f"part_A/{name}" for name in base] + base
    if dataset_name == "shb":
        return [f"part_B/{name}" for name in base] + base
    return base


def _find_existing_dir(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find any of these directories under {root}: {candidates}"
    )


def _find_child_dir(base: Path, names: tuple[str, ...]) -> Path:
    direct_children = {child.name.lower(): child for child in base.iterdir() if child.is_dir()}
    for name in names:
        path = direct_children.get(name.lower())
        if path is not None:
            return path
    return base


def _split_manifest_candidates(root: Path, split: str) -> list[Path]:
    names = [split, split.lower(), split.capitalize()]
    candidates = []
    for name in dict.fromkeys(names):
        candidates.extend(
            [
                root / f"{name}.txt",
                root / "list" / f"{name}.txt",
                root / "lists" / f"{name}.txt",
                root / "image_list" / f"{name}.txt",
                root / "image_lists" / f"{name}.txt",
            ]
        )
    return candidates


def _find_split_manifest(root: Path, split: str) -> Optional[Path]:
    for candidate in _split_manifest_candidates(root, split):
        if candidate.is_file():
            return candidate
    return None


def _read_manifest_stems(manifest_path: Path) -> set[str]:
    stems = set()
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.replace(",", " ").split()[0]
        token_path = Path(token)
        stems.add(token)
        stems.add(token_path.name)
        stems.add(token_path.stem)
    return stems


def _image_matches_manifest(image_path: Path, images_dir: Path, manifest_stems: set[str]) -> bool:
    rel_path = image_path.relative_to(images_dir).as_posix()
    return (
        rel_path in manifest_stems
        or Path(rel_path).with_suffix("").as_posix() in manifest_stems
        or image_path.name in manifest_stems
        or image_path.stem in manifest_stems
    )


def _collect_images(images_dir: Path, manifest_stems: Optional[set[str]] = None) -> list[Path]:
    image_files = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if manifest_stems is not None:
        image_files = [
            path
            for path in image_files
            if _image_matches_manifest(path, images_dir, manifest_stems)
        ]
    if not image_files:
        raise RuntimeError(f"No images found in {images_dir}")
    return image_files


def _mat_candidates(image_path: Path, gt_dir: Path) -> list[Path]:
    stem = image_path.stem
    return [
        gt_dir / f"GT_{stem}.mat",
        gt_dir / f"{stem}.mat",
        gt_dir / f"{stem}_ann.mat",
        gt_dir / f"{stem}_gt.mat",
        gt_dir / f"{stem}_points.mat",
    ]


def _text_candidates(image_path: Path, gt_dir: Path) -> list[Path]:
    stem = image_path.stem
    return [
        gt_dir / f"{stem}.txt",
        gt_dir / f"{stem}.csv",
        gt_dir / f"{stem}.json",
        gt_dir / f"GT_{stem}.txt",
        gt_dir / f"GT_{stem}.csv",
        gt_dir / f"GT_{stem}.json",
        gt_dir / f"{stem}_ann.txt",
        gt_dir / f"{stem}_ann.csv",
        gt_dir / f"{stem}_ann.json",
    ]


def _extract_points_from_object(value) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.flat:
                points = _extract_points_from_object(item)
                if points is not None:
                    return points
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
        if arr.ndim == 2 and arr.shape[0] >= 2:
            return arr[:2, :].T
    return None


def load_points(annotation_path: Path) -> np.ndarray:
    if annotation_path.suffix.lower() == ".mat":
        mat = scipy.io.loadmat(annotation_path)
        for key in POINT_KEYS:
            if key in mat:
                points = _extract_points_from_object(mat[key])
                if points is not None:
                    return points.astype(np.float32)
        raise KeyError(
            f"No point array found in {annotation_path}. Tried keys: {POINT_KEYS}"
        )

    if annotation_path.suffix.lower() == ".json":
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for key in POINT_KEYS:
            if isinstance(data, dict) and key in data:
                points = _extract_points_from_object(np.asarray(data[key]))
                if points is not None:
                    return points.astype(np.float32)
        points = _extract_points_from_object(np.asarray(data))
        if points is not None:
            return points.astype(np.float32)
        raise KeyError(
            f"No point array found in {annotation_path}. Tried keys: {POINT_KEYS}"
        )

    if annotation_path.stat().st_size == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.loadtxt(
        annotation_path,
        delimiter="," if annotation_path.suffix.lower() == ".csv" else None,
    )
    arr = np.atleast_2d(arr).astype(np.float32)
    if arr.shape[1] < 2:
        raise ValueError(f"Annotation file must have at least two columns: {annotation_path}")
    return arr[:, :2]


class CrowdCountingDataset(Dataset):
    """
    Generic crowd-counting dataset that turns point annotations into density maps.

    Supported layout examples:
      data/ShanghaiTech/part_A/train_data/{images,ground_truth}
      data/UCF-QNRF/train/{images,ground_truth}
      data/NWPU/train/{images,gt}
      data/JHU/train/{images,ground_truth}
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "sha",
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        input_size: Tuple[int, int] = (384, 384),
        sigma: float = 5.0,
        density_map_size: Optional[Tuple[int, int]] = None,
        return_count: bool = False,
    ):
        self.root = Path(root)
        self.dataset_name = _normalise_dataset_name(dataset_name)
        self.split = split
        self.transform = transform or get_default_transform(input_size)
        self.input_size = input_size
        self.sigma = sigma
        self.density_map_size = density_map_size or input_size
        self.return_count = return_count

        manifest_path = _find_split_manifest(self.root, split)
        manifest_stems = _read_manifest_stems(manifest_path) if manifest_path else None
        try:
            split_dir = _find_existing_dir(
                self.root, _candidate_split_names(self.dataset_name, split)
            )
        except FileNotFoundError:
            if manifest_path is None:
                raise
            split_dir = self.root
        self.images_dir = _find_child_dir(split_dir, ("images", "img", "imgs"))
        self.gt_dirs = [
            _find_child_dir(
                split_dir,
                (
                    "ground_truth",
                    "gt",
                    "gt_mat",
                    "annotations",
                    "ann",
                    "mats",
                    "jsons",
                    "labels",
                ),
            )
        ]
        if split_dir == self.root:
            for name in (
                "ground_truth",
                "gt",
                "gt_mat",
                "annotations",
                "ann",
                "mats",
                "jsons",
                "labels",
            ):
                candidate = _find_child_dir(self.root, (name,))
                if candidate not in self.gt_dirs:
                    self.gt_dirs.append(candidate)
        self.image_files = _collect_images(
            self.images_dir, manifest_stems=manifest_stems
        )

    @staticmethod
    def _load_image(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _scale_points(
        points: np.ndarray,
        orig_wh: Tuple[int, int],
        to_wh: Tuple[int, int],
    ) -> np.ndarray:
        if points.size == 0:
            return points.reshape(0, 2)
        ow, oh = orig_wh
        tw, th = to_wh
        sx, sy = tw / ow, th / oh
        return np.stack([points[:, 0] * sx, points[:, 1] * sy], axis=1)

    def _annotation_path(self, image_path: Path) -> Path:
        for gt_dir in self.gt_dirs:
            for candidate in _mat_candidates(image_path, gt_dir) + _text_candidates(
                image_path, gt_dir
            ):
                if candidate.is_file():
                    return candidate
        raise FileNotFoundError(f"No annotation file found for {image_path}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        img = self._load_image(img_path)
        orig_wh = img.size
        img = img.resize(self.input_size, Image.Resampling.BILINEAR)

        img_tensor = self.transform(img)
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)

        points = load_points(self._annotation_path(img_path))
        scaled_for_density = self._scale_points(
            points, orig_wh, self.density_map_size
        )
        density_map = create_density_map(
            centroids=scaled_for_density,
            img_size=self.density_map_size,
            sigma=self.sigma,
        )

        if self.return_count:
            return img_tensor, torch.tensor([density_map.sum()], dtype=torch.float32)

        den_tensor = torch.from_numpy(density_map).unsqueeze(0)
        return img_tensor, den_tensor


class ShanghaiTechDataset(CrowdCountingDataset):
    def __init__(self, root: str, part: str = "part_A", split: str = "train_data", **kwargs):
        dataset_name = "sha" if part in {"part_A", "A", "sha"} else "shb"
        split_name = "train" if split == "train_data" else "test" if split == "test_data" else split
        super().__init__(
            root=root,
            dataset_name=dataset_name,
            split=split_name,
            **kwargs,
        )


class CrowdCountingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_folder,
        dataset_name="sha",
        validation_split=0.1,
        seed=42,
        sigma=5,
        return_count=False,
        batch_size=8,
        num_workers=4,
        device=None,
        input_size=(384, 384),
        density_map_size=(384, 384),
        train_split="train",
        test_split="test",
        eval_data_folder=None,
        eval_dataset_name=None,
        eval_split=None,
    ):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.eval_data_folder = eval_data_folder or data_folder
        self.eval_dataset_name = eval_dataset_name or dataset_name
        self.train_split = train_split
        self.test_split = test_split
        self.eval_split = eval_split or test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.seed = seed
        self.sigma = sigma
        self.return_count = return_count
        self.input_size = input_size
        self.density_map_size = density_map_size
        self.device = device or get_device()
        self.pin_memory = True if self.device != "mps" else False
        self.transform = get_default_transform(input_size=self.input_size)

    def _make_dataset(self, root, dataset_name, split):
        return CrowdCountingDataset(
            root=root,
            dataset_name=dataset_name,
            split=split,
            transform=self.transform,
            input_size=self.input_size,
            sigma=self.sigma,
            density_map_size=self.density_map_size,
            return_count=self.return_count,
        )

    def setup(self, stage=None):
        full_dataset = self._make_dataset(
            self.data_folder,
            self.dataset_name,
            self.train_split,
        )

        train_size = int(len(full_dataset) * (1 - self.validation_split))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.test_dataset = self._make_dataset(
            self.eval_data_folder,
            self.eval_dataset_name,
            self.eval_split,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )


class ShanghaiTechDataModule(CrowdCountingDataModule):
    def __init__(self, data_folder, part="part_A", **kwargs):
        dataset_name = "sha" if part in {"part_A", "A", "sha"} else "shb"
        super().__init__(
            data_folder=data_folder,
            dataset_name=dataset_name,
            train_split="train",
            test_split="test",
            **kwargs,
        )


if __name__ == "__main__":
    data_module = CrowdCountingDataModule(
        data_folder="./data/ShanghaiTech",
        dataset_name="sha",
        batch_size=3,
        validation_split=0.1,
        sigma=5,
        density_map_size=(384, 384),
        input_size=(384, 384),
        return_count=False,
    )
    show_samples_from_loaders(data_module)
