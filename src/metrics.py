import torch
import torch.nn.functional as F


def pixelwise_mse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pixel-wise Mean Squared Error (MSE) between pred and target density maps.

    Args:
        pred (Tensor): Predicted density maps, shape (B, H, W) or (B, 1, H, W).
        target (Tensor): Ground truth density maps, same shape as pred.
        reduction (str): 'mean' | 'sum' | 'none'.

    Returns:
        Tensor: MSE loss. If 'none', returns per-sample MSE of shape (B,). Otherwise, scalar.
    """
    return F.mse_loss(pred, target, reduction=reduction)


def pixelwise_mae(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pixel-wise Mean Absolute Error (MAE).
    """
    return F.l1_loss(pred, target, reduction=reduction)


def pixelwise_rmse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute pixel-wise Root Mean Squared Error (RMSE).
    """
    mse = pixelwise_mse(pred, target, reduction)
    return torch.sqrt(mse)


def spatial_count(density_map: torch.Tensor) -> torch.Tensor:
    """
    Sum the density map across spatial dimensions to get counts per sample.

    Args:
        density_map (Tensor): Shape (B, H, W) or (B, 1, H, W).

    Returns:
        Tensor: Counts per sample, shape (B,).
    """

    # If channel dimension exists, remove it
    if density_map.ndim == 4 and density_map.shape[1] == 1:
        x = density_map.squeeze(1)
    else:
        x = density_map
    # sum over spatial dims
    # assumes x.shape = (B, H, W)
    return x.view(x.size(0), -1).sum(dim=1)  # (B,)


def count_mae(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute error in total counts between pred and target.
    """
    c_pred = spatial_count(pred)
    c_gt = spatial_count(target)
    return F.l1_loss(c_pred, c_gt, reduction=reduction)


# New functions for count MSE and RMSE
def count_mse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Mean Squared Error (MSE) of total counts between pred and target.
    """
    c_pred = spatial_count(pred)
    c_gt = spatial_count(target)
    return F.mse_loss(c_pred, c_gt, reduction=reduction)


def count_rmse(
    pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute Root Mean Squared Error (RMSE) of total counts between pred and target.
    """
    mse = count_mse(pred, target, reduction)
    return torch.sqrt(mse)


def empty_region_fp_mass(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 1e-5
) -> torch.Tensor:
    """
    Compute total predicted density where the ground-truth density is essentially zero.
    Returns the average false-positive mass per sample in the batch.
    """
    mask = target < threshold
    fp_mass = (pred * mask).view(pred.size(0), -1).sum(dim=1)
    return fp_mass.mean()


def dense_region_mse(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 1e-3
) -> torch.Tensor:
    """
    Compute MSE specifically in dense regions (where target >= threshold)
    to evaluate peak preservation and sharpness.
    """
    mask = target >= threshold
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask], target[mask], reduction="mean")


def game(pred: torch.Tensor, target: torch.Tensor, level: int = 0) -> torch.Tensor:
    """
    Compute Grid Average Mean Absolute Error (GAME) at a specific level.
    GAME(0) is equivalent to Count MAE.
    GAME(1) divides the image into 4 patches (2x2), GAME(2) into 16 (4x4), etc.
    """
    if level == 0:
        return count_mae(pred, target, reduction="mean")
    
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
        
    grid_h = 2 ** level
    grid_w = 2 ** level
    
    B, C, H, W = pred.shape
    patch_h = H // grid_h
    patch_w = W // grid_w
    
    # Crop to ensure perfect divisibility
    h_crop = patch_h * grid_h
    w_crop = patch_w * grid_w
    
    p = pred[:, :, :h_crop, :w_crop]
    t = target[:, :, :h_crop, :w_crop]
    
    # Compute sums within each grid cell
    p_sum = p.view(B, C, grid_h, patch_h, grid_w, patch_w).sum(dim=(3, 5))
    t_sum = t.view(B, C, grid_h, patch_h, grid_w, patch_w).sum(dim=(3, 5))
    
    # GAME is the sum of absolute errors in grid cells
    return torch.abs(p_sum - t_sum).sum(dim=(1, 2, 3)).mean()


if __name__ == "__main__":
    # Example usage
    pred = torch.randn(4, 1, 384, 384)
    target = torch.randn(4, 1, 384, 384)

    print("Pixelwise MSE:", pixelwise_mse(pred, target))
    print("Pixelwise MAE:", pixelwise_mae(pred, target))
    print("Pixelwise RMSE:", pixelwise_rmse(pred, target))
    print("Spatial Count:", spatial_count(pred))
