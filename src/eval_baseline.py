from typing import Callable, Dict, Optional, Tuple

import torch

from src.metrics import (
    count_mae,
    count_mse,
    pixelwise_mae,
    pixelwise_mse,
    spatial_count,
)


def compute_baseline_metrics(
    dt_loader: torch.utils.data.DataLoader,
    device: torch.device,
    baseline_fn: Callable[[torch.Tensor], torch.Tensor],
) -> Tuple[float, float, float, float]:
    """
    Compute count and pixel metrics for a density-map baseline.

    Count MAE/RMSE are averaged over samples. Pixel MAE/RMSE are computed
    globally over all pixels, so uneven final batch sizes cannot skew results.
    """
    total_mae = 0.0
    total_mse = 0.0
    total_pixel_abs = 0.0
    total_pixel_sq = 0.0

    total_samples = 0
    total_pixels = 0

    with torch.no_grad():
        for _, gt_density in dt_loader:
            gt_density = gt_density.to(device)
            batch_size = gt_density.size(0)
            total_samples += batch_size
            pred_density = baseline_fn(gt_density)

            # Compute MAE and MSE across counts using shared metrics
            batch_mae = count_mae(pred_density, gt_density, reduction="sum").item()
            batch_mse = count_mse(pred_density, gt_density, reduction="sum").item()

            # Pixel-wise totals. Final MAE/RMSE are normalized once at the end.
            pixel_abs = pixelwise_mae(pred_density, gt_density, reduction="sum").item()
            pixel_sq = pixelwise_mse(pred_density, gt_density, reduction="sum").item()

            total_mae += batch_mae
            total_mse += batch_mse
            total_pixel_abs += pixel_abs
            total_pixel_sq += pixel_sq
            total_pixels += gt_density.numel()

    if total_samples == 0:
        raise ValueError("Cannot evaluate metrics on an empty dataloader")

    mae = total_mae / total_samples
    rmse = (total_mse / total_samples) ** 0.5
    pixel_mae = total_pixel_abs / total_pixels
    pixel_rmse = (total_pixel_sq / total_pixels) ** 0.5
    return mae, rmse, pixel_mae, pixel_rmse


def evaluate_baselines(
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_pixels: Optional[int] = None,
    print_results: bool = True,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Evaluate a set of baseline predictors in one shot.
    Returns a dict mapping baseline name to:
    (count MAE, count RMSE, pixel MAE, pixel RMSE).
    """
    # 1) Compute the average count per image over the entire set
    total_count = 0.0
    total_samples = 0
    with torch.no_grad():
        for _, gt_density in data_loader:
            gt_density = gt_density.to(device)
            if n_pixels is None:
                n_pixels = gt_density[0].numel()
            # spatial_count returns a Tensor of shape (batch,)
            total_count += spatial_count(gt_density).sum().item()
            total_samples += gt_density.size(0)

    if total_samples == 0:
        raise ValueError("Cannot evaluate baselines on an empty dataloader")
    if n_pixels is None:
        raise ValueError("Could not infer density-map size from dataloader")

    # derive per-pixel density so that sum = mean_count per image
    mean_count = total_count / total_samples
    mean_density_pixel = mean_count / n_pixels

    # 2) Define each baseline as a function gt_density -> pred_density
    baselines: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "mean_count": lambda gt: torch.full_like(gt, mean_density_pixel),
        "zeros": lambda gt: torch.zeros_like(gt),
    }

    # 3) Compute and print metrics for each
    results = {}
    for name, fn in baselines.items():
        mae, rmse, pixel_mae, pixel_rmse = compute_baseline_metrics(
            data_loader, device, fn
        )
        results[name] = (mae, rmse, pixel_mae, pixel_rmse)
        if print_results:
            print(
                f"{name:10s}: Count MAE = {mae:.3f}, Count RMSE = {rmse:.6f}, "
                f"Pixel MAE = {pixel_mae:.6f}, Pixel RMSE = {pixel_rmse:.6f}"
            )

    return results


if __name__ == "__main__":
    import torch

    from src.data_loader import CrowdCountingDataModule
    from src.utils import get_device

    # Initialize device
    device = get_device()

    # Load the data
    data_module = CrowdCountingDataModule(
        data_folder="./data/ShanghaiTech",
        dataset_name="sha",
        validation_split=0.1,
        sigma=5,
        return_count=False,
        batch_size=8,
        num_workers=4,
        input_size=(384, 384),
        density_map_size=(192, 192),
        device=device,
    )
    data_module.setup()

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    n_pixels = 192 * 192
    print("\n=== Baseline Results on TRAIN ===")
    evaluate_baselines(train_loader, device, n_pixels)

    print("\n=== Baseline Results on VALIDATION ===")
    evaluate_baselines(val_loader, device, n_pixels)

    print("\n=== Baseline Results on TEST ===")
    evaluate_baselines(test_loader, device, n_pixels)
