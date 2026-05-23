import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_loader import (
    AugmentedCrowdCountingDataset,
    CropAugmentationConfig,
    CrowdCountingDataset,
    _count_bin,
    _nearest_neighbor_distances,
)
from src.utils import create_density_map


COUNT_BINS = [0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, float("inf")]


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected size as WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def parse_range(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected range as MIN,MAX")
    return parts[0], parts[1]


def density_occupied_fraction(
    points: np.ndarray,
    orig_wh: tuple[int, int],
    density_map_size: tuple[int, int],
    sigma: float,
    threshold: float,
) -> float:
    if points.size == 0:
        return 0.0
    ow, oh = orig_wh
    dw, dh = density_map_size
    scaled = np.stack([points[:, 0] * dw / ow, points[:, 1] * dh / oh], axis=1)
    density_map = create_density_map(scaled, img_size=density_map_size, sigma=sigma)
    return float((density_map > threshold).mean())


def original_record(
    dataset: CrowdCountingDataset,
    idx: int,
    density_map_size: tuple[int, int],
    sigma: float,
    threshold: float,
) -> dict:
    img, points, img_path = dataset.load_raw_sample(idx)
    width, height = img.size
    area = max(width * height, 1)
    nn_distances = _nearest_neighbor_distances(points)
    count = float(len(points))
    return {
        "kind": "original",
        "image": str(img_path),
        "width": width,
        "height": height,
        "area": area,
        "count": count,
        "density_per_megapixel": float(count / area * 1_000_000),
        "point_occupancy": float(count / area),
        "density_occupied_fraction": density_occupied_fraction(
            points,
            (width, height),
            density_map_size,
            sigma,
            threshold,
        ),
        "mean_nn_distance": float(nn_distances.mean()) if len(nn_distances) else None,
        "median_nn_distance": float(np.median(nn_distances)) if len(nn_distances) else None,
        "count_bin": _count_bin(count),
    }


def crop_record(
    dataset: CrowdCountingDataset,
    sampler: AugmentedCrowdCountingDataset,
    idx: int,
    crop_slot: int,
    density_map_size: tuple[int, int],
    sigma: float,
    threshold: float,
) -> dict:
    metadata = sampler.sample_crop_metadata(idx, crop_slot=crop_slot)
    img, points, _ = dataset.load_raw_sample(idx)
    x0, y0, x1, y1 = metadata["crop_box"]
    crop_points = AugmentedCrowdCountingDataset._points_in_box(points, metadata["crop_box"])
    metadata["kind"] = "crop"
    metadata["point_occupancy"] = metadata.pop("occupancy")
    metadata["density_occupied_fraction"] = density_occupied_fraction(
        crop_points,
        (x1 - x0, y1 - y0),
        density_map_size,
        sigma,
        threshold,
    )
    return metadata


def summarize(frame: pd.DataFrame) -> dict:
    count_hist = pd.cut(
        frame["count"],
        bins=COUNT_BINS,
        right=False,
        include_lowest=True,
    ).value_counts(sort=False)
    summary = {
        "samples": int(len(frame)),
        "empty_fraction": float((frame["count"] <= 0).mean()),
        "near_empty_fraction": float((frame["count"] <= 2).mean()),
        "count": frame["count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "density_per_megapixel": frame["density_per_megapixel"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_dict(),
        "density_occupied_fraction": frame["density_occupied_fraction"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_dict(),
        "mean_nn_distance": frame["mean_nn_distance"]
        .dropna()
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .to_dict(),
        "count_histogram": {str(key): int(value) for key, value in count_hist.items()},
        "count_bins": frame["count_bin"].value_counts().to_dict(),
    }
    return summary


def plot_histograms(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for column, xlabel in [
        ("count", "count"),
        ("density_per_megapixel", "heads per megapixel"),
        ("density_occupied_fraction", "density occupied fraction"),
        ("mean_nn_distance", "mean nearest-neighbor distance"),
    ]:
        plt.figure(figsize=(8, 4))
        for kind, group in frame.groupby("kind"):
            values = group[column].dropna()
            if column in {"count", "density_per_megapixel"}:
                values = np.log1p(values)
                label = f"{kind} (log1p)"
            else:
                label = kind
            plt.hist(values, bins=30, alpha=0.55, label=label)
        plt.xlabel(xlabel)
        plt.ylabel("samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}.png", dpi=150)
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare full-image and crop augmentation data distributions."
    )
    parser.add_argument("--data-folder", default="./data/ShanghaiTech")
    parser.add_argument("--dataset", default="sha")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--crops-per-image", type=int, default=2)
    parser.add_argument("--crop-size", type=parse_size, default=(256, 256))
    parser.add_argument("--scale-jitter", type=parse_range, default=(0.75, 1.25))
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--density-map-size", type=parse_size, default=(128, 128))
    parser.add_argument("--density-threshold", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="./outputs/augmentation_distribution.json")
    parser.add_argument("--plot-dir", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = CrowdCountingDataset(
        root=args.data_folder,
        dataset_name=args.dataset,
        split=args.split,
        sigma=args.sigma,
    )
    sample_count = min(args.max_images, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(len(dataset), size=sample_count, replace=False))

    crop_config = CropAugmentationConfig(
        enabled=True,
        crop_size=args.crop_size,
        crops_per_image=args.crops_per_image,
        full_image_probability=0.0,
        scale_jitter=args.scale_jitter,
        photometric_jitter=0.0,
    )
    sampler = AugmentedCrowdCountingDataset(dataset, crop_config, seed=args.seed)

    records = []
    for idx in indices:
        records.append(
            original_record(
                dataset,
                int(idx),
                args.density_map_size,
                args.sigma,
                args.density_threshold,
            )
        )
        for crop_slot in range(args.crops_per_image):
            records.append(
                crop_record(
                    dataset,
                    sampler,
                    int(idx),
                    crop_slot,
                    args.density_map_size,
                    args.sigma,
                    args.density_threshold,
                )
            )

    frame = pd.DataFrame.from_records(records)
    output = {
        "config": vars(args),
        "original": summarize(frame[frame["kind"] == "original"]),
        "crop": summarize(frame[frame["kind"] == "crop"]),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    frame.to_csv(output_path.with_suffix(".csv"), index=False)
    if args.plot_dir:
        plot_histograms(frame, Path(args.plot_dir))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
