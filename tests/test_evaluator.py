import math
import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.eval_baseline import compute_baseline_metrics, evaluate_baselines
from src.utils import resize_density_map_count_preserving


def make_loader(density_maps: torch.Tensor, batch_size: int = 2) -> DataLoader:
    images = torch.zeros((density_maps.shape[0], 3, 2, 2), dtype=torch.float32)
    return DataLoader(TensorDataset(images, density_maps), batch_size=batch_size)


class EvaluatorTests(unittest.TestCase):
    def test_single_nonzero_pixel_metrics_are_analytically_known(self):
        density_maps = torch.tensor(
            [[[[3.0, 0.0], [0.0, 0.0]]]],
            dtype=torch.float32,
        )
        loader = make_loader(density_maps, batch_size=1)

        metrics = compute_baseline_metrics(
            loader,
            torch.device("cpu"),
            baseline_fn=lambda gt: torch.zeros_like(gt),
        )

        count_mae, count_rmse, pixel_mae, pixel_rmse = metrics
        self.assertAlmostEqual(count_mae, 3.0)
        self.assertAlmostEqual(count_rmse, 3.0)
        self.assertAlmostEqual(pixel_mae, 3.0 / 4.0)
        self.assertAlmostEqual(pixel_rmse, math.sqrt(9.0 / 4.0))
        self.assertAlmostEqual(empty_fp, 0.0)

    def test_zero_baseline_metrics_are_analytically_known(self):
        density_maps = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [2.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        loader = make_loader(density_maps, batch_size=2)

        results = evaluate_baselines(
            loader, torch.device("cpu"), n_pixels=4, print_results=False
        )
        count_mae, count_rmse, pixel_mae, pixel_rmse, game_3, empty_fp = results["zeros"]

        self.assertAlmostEqual(count_mae, 2.0)
        self.assertAlmostEqual(count_rmse, math.sqrt(5.0))
        self.assertAlmostEqual(pixel_mae, 4.0 / 8.0)
        self.assertAlmostEqual(pixel_rmse, math.sqrt(6.0 / 8.0))
        self.assertAlmostEqual(empty_fp, 0.0)

    def test_mean_baseline_metrics_are_analytically_known(self):
        density_maps = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 1.0], [2.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        loader = make_loader(density_maps, batch_size=2)

        results = evaluate_baselines(
            loader, torch.device("cpu"), n_pixels=4, print_results=False
        )
        count_mae, count_rmse, pixel_mae, pixel_rmse, game_3, empty_fp = results["mean_count"]

        self.assertAlmostEqual(count_mae, 1.0)
        self.assertAlmostEqual(count_rmse, 1.0)
        self.assertAlmostEqual(pixel_mae, 5.0 / 8.0)
        self.assertAlmostEqual(pixel_rmse, math.sqrt(4.0 / 8.0))
        self.assertAlmostEqual(empty_fp, 0.0)

    def test_metrics_are_weighted_correctly_with_uneven_batches(self):
        density_maps = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 4.0]]],
            ],
            dtype=torch.float32,
        )
        loader = make_loader(density_maps, batch_size=2)

        metrics = compute_baseline_metrics(
            loader,
            torch.device("cpu"),
            baseline_fn=lambda gt: torch.zeros_like(gt),
        )

        count_mae, count_rmse, pixel_mae, pixel_rmse, game_3, empty_fp = metrics
        self.assertAlmostEqual(count_mae, 5.0 / 3.0)
        self.assertAlmostEqual(count_rmse, math.sqrt(17.0 / 3.0))
        self.assertAlmostEqual(pixel_mae, 5.0 / 12.0)
        self.assertAlmostEqual(pixel_rmse, math.sqrt(17.0 / 12.0))
        self.assertAlmostEqual(empty_fp, 0.0)


class DensityResizeTests(unittest.TestCase):
    def test_resize_density_map_preserves_sum_when_count_preservation_is_required(self):
        density_map = torch.tensor(
            [[[[1.0, 0.0], [0.0, 3.0]]]],
            dtype=torch.float32,
        )

        upsampled = resize_density_map_count_preserving(density_map, size=(4, 4))
        downsampled = resize_density_map_count_preserving(density_map, size=(1, 1))

        self.assertAlmostEqual(
            upsampled.sum().item(), density_map.sum().item(), places=6
        )
        self.assertAlmostEqual(
            downsampled.sum().item(), density_map.sum().item(), places=6
        )

    def test_resize_density_map_preserves_sum_for_non_square_sizes(self):
        density_map = torch.arange(30, dtype=torch.float32).reshape(2, 1, 3, 5)

        resized = resize_density_map_count_preserving(density_map, size=(7, 4))

        original_sums = density_map.sum(dim=(-2, -1))
        resized_sums = resized.sum(dim=(-2, -1))
        torch.testing.assert_close(resized_sums, original_sums)


if __name__ == "__main__":
    unittest.main()
