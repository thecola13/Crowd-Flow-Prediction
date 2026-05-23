import json
import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CropAugmentationConfig:
    enabled: bool = False
    crop_size: Tuple[int, int] = (256, 256)
    crops_per_image: int = 1
    full_image_probability: float = 0.5
    empty_count_threshold: float = 0.5
    near_empty_count_threshold: float = 2.0
    scale_jitter: Tuple[float, float] = (0.75, 1.25)
    horizontal_flip_probability: float = 0.5
    photometric_jitter: float = 0.15
    max_crop_resample_attempts: int = 30


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
    }
    key = dataset_name.lower()
    if key not in aliases:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Supported values include sha, shb, and qnrf."
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


def _count_bin(count: float, near_empty_threshold: float = 2.0) -> str:
    if count <= 0:
        return "empty"
    if count <= near_empty_threshold:
        return "near_empty"
    if count <= 20:
        return "low"
    if count <= 100:
        return "medium"
    return "high"


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0,), dtype=np.float32)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1).astype(np.float32)


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
    if isinstance(value, (np.ndarray, np.void)):
        if value.dtype.names is not None:
            # Handle structured array/void
            for key in ("location", "points", "point", "locations", "annPoints"):
                if key in value.dtype.names:
                    points = _extract_points_from_object(value[key])
                    if points is not None:
                        return points
            for name in value.dtype.names:
                points = _extract_points_from_object(value[name])
                if points is not None:
                    return points
            return None

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
        self._points_cache: dict[Path, np.ndarray] = {}

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

    def _load_points_for_image(self, image_path: Path) -> np.ndarray:
        if image_path not in self._points_cache:
            self._points_cache[image_path] = load_points(self._annotation_path(image_path))
        return self._points_cache[image_path]

    def _image_to_tensor(self, img: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(img)
        if not isinstance(img_tensor, torch.Tensor):
            img_tensor = torch.tensor(img_tensor, dtype=torch.float32)
        return img_tensor

    def _sample_to_tensors(
        self,
        img: Image.Image,
        points: np.ndarray,
        orig_wh: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img = img.resize(self.input_size, Image.Resampling.BILINEAR)
        img_tensor = self._image_to_tensor(img)
        scaled_for_density = self._scale_points(
            points,
            orig_wh,
            self.density_map_size,
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

    def load_raw_sample(self, idx: int) -> tuple[Image.Image, np.ndarray, Path]:
        img_path = self.image_files[idx]
        img = self._load_image(img_path)
        points = self._load_points_for_image(img_path)
        return img, points.copy(), img_path

    def get_sample_metadata(self, idx: int) -> dict:
        img, points, img_path = self.load_raw_sample(idx)
        nn_distances = _nearest_neighbor_distances(points)
        width, height = img.size
        area = max(width * height, 1)
        return {
            "image": str(img_path),
            "width": width,
            "height": height,
            "area": area,
            "count": float(len(points)),
            "density_per_megapixel": float(len(points) / area * 1_000_000),
            "occupancy": float(len(points) / area),
            "mean_nn_distance": float(nn_distances.mean()) if len(nn_distances) else None,
            "median_nn_distance": float(np.median(nn_distances)) if len(nn_distances) else None,
            "count_bin": _count_bin(float(len(points))),
        }

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, points, _ = self.load_raw_sample(idx)
        return self._sample_to_tensors(img, points, img.size)


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


class AugmentedCrowdCountingDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        config: CropAugmentationConfig,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.config = config
        self.seed = seed
        self.color_jitter = (
            transforms.ColorJitter(
                brightness=config.photometric_jitter,
                contrast=config.photometric_jitter,
                saturation=config.photometric_jitter,
                hue=min(config.photometric_jitter / 3, 0.05),
            )
            if config.photometric_jitter > 0
            else None
        )

    def __len__(self) -> int:
        return len(self.dataset) * max(1, self.config.crops_per_image)

    def _resolve_base(self, idx: int) -> tuple[CrowdCountingDataset, int]:
        dataset = self.dataset
        while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            idx = dataset.indices[idx]
            dataset = dataset.dataset
        if not isinstance(dataset, CrowdCountingDataset):
            raise TypeError(
                "AugmentedCrowdCountingDataset expects a CrowdCountingDataset "
                "or a Subset wrapping one."
            )
        return dataset, idx

    def _crop_box(
        self,
        width: int,
        height: int,
        rng: np.random.Generator,
    ) -> tuple[int, int, int, int]:
        crop_w, crop_h = self.config.crop_size
        scale_min, scale_max = self.config.scale_jitter
        scale = float(rng.uniform(scale_min, scale_max))
        source_w = min(width, max(1, int(round(crop_w / scale))))
        source_h = min(height, max(1, int(round(crop_h / scale))))
        x0 = int(rng.integers(0, max(width - source_w + 1, 1)))
        y0 = int(rng.integers(0, max(height - source_h + 1, 1)))
        return x0, y0, x0 + source_w, y0 + source_h

    @staticmethod
    def _points_in_box(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = box
        if points.size == 0:
            return points.reshape(0, 2)
        keep = (
            (points[:, 0] >= x0)
            & (points[:, 0] < x1)
            & (points[:, 1] >= y0)
            & (points[:, 1] < y1)
        )
        cropped = points[keep].copy()
        cropped[:, 0] -= x0
        cropped[:, 1] -= y0
        return cropped

    def sample_crop_metadata(
        self,
        dataset_idx: int,
        crop_slot: int = 0,
    ) -> dict:
        base, base_idx = self._resolve_base(dataset_idx)
        img, points, img_path = base.load_raw_sample(base_idx)
        width, height = img.size
        rng = np.random.default_rng(self.seed + base_idx * 1009 + crop_slot * 9176)
        target_bins = ("empty", "near_empty", "low", "medium", "high")
        target_bin = target_bins[(base_idx + crop_slot) % len(target_bins)]
        selected_box = (0, 0, width, height)
        selected_points = points
        selected_bin = _count_bin(float(len(points)), self.config.near_empty_count_threshold)

        for _ in range(max(1, self.config.max_crop_resample_attempts)):
            box = self._crop_box(width, height, rng)
            crop_points = self._points_in_box(points, box)
            crop_bin = _count_bin(
                float(len(crop_points)),
                self.config.near_empty_count_threshold,
            )
            selected_box = box
            selected_points = crop_points
            selected_bin = crop_bin
            if crop_bin == target_bin:
                break

        x0, y0, x1, y1 = selected_box
        crop_w, crop_h = x1 - x0, y1 - y0
        area = max(crop_w * crop_h, 1)
        nn_distances = _nearest_neighbor_distances(selected_points)
        return {
            "image": str(img_path),
            "source_index": base_idx,
            "crop_box": selected_box,
            "width": crop_w,
            "height": crop_h,
            "area": area,
            "count": float(len(selected_points)),
            "density_per_megapixel": float(len(selected_points) / area * 1_000_000),
            "occupancy": float(len(selected_points) / area),
            "mean_nn_distance": float(nn_distances.mean()) if len(nn_distances) else None,
            "median_nn_distance": float(np.median(nn_distances)) if len(nn_distances) else None,
            "count_bin": selected_bin,
            "target_bin": target_bin,
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_idx = idx % len(self.dataset)
        crop_slot = idx // len(self.dataset)
        base, base_idx = self._resolve_base(dataset_idx)
        img, points, _ = base.load_raw_sample(base_idx)

        if np.random.random() < self.config.full_image_probability:
            return base._sample_to_tensors(img, points, img.size)

        metadata = self.sample_crop_metadata(dataset_idx, crop_slot)
        x0, y0, x1, y1 = metadata["crop_box"]
        cropped_img = img.crop((x0, y0, x1, y1))
        cropped_points = self._points_in_box(points, metadata["crop_box"])

        if np.random.random() < self.config.horizontal_flip_probability:
            width = cropped_img.size[0]
            cropped_img = cropped_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if cropped_points.size:
                cropped_points[:, 0] = width - 1 - cropped_points[:, 0]

        cropped_img = cropped_img.resize(base.input_size, Image.Resampling.BILINEAR)
        if self.color_jitter is not None:
            cropped_img = self.color_jitter(cropped_img)
        img_tensor = base._image_to_tensor(cropped_img)

        scaled_points = base._scale_points(
            cropped_points,
            (x1 - x0, y1 - y0),
            base.density_map_size,
        )
        density_map = create_density_map(
            centroids=scaled_points,
            img_size=base.density_map_size,
            sigma=base.sigma,
        )
        if base.return_count:
            return img_tensor, torch.tensor([density_map.sum()], dtype=torch.float32)
        return img_tensor, torch.from_numpy(density_map).unsqueeze(0)


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
        crop_augmentation: Optional[CropAugmentationConfig] = None,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.eval_data_folder = eval_data_folder or data_folder
        self.eval_dataset_name = eval_dataset_name or dataset_name
        self.train_split = train_split
        self.test_split = test_split
        self.eval_split = eval_split or test_split
        self.crop_augmentation = crop_augmentation or CropAugmentationConfig()
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
        if self.crop_augmentation.enabled:
            self.train_dataset = AugmentedCrowdCountingDataset(
                self.train_dataset,
                self.crop_augmentation,
                seed=self.seed,
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
