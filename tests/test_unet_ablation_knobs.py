import unittest

import torch

from src.models.unet import UNet


class UNetAblationKnobTests(unittest.TestCase):
    def test_before_pool_unet_supports_full_half_and_quarter_outputs(self):
        x = torch.randn(2, 3, 64, 64)

        for output_reduction, expected_size in [(1, 64), (2, 32), (4, 16)]:
            model = UNet(
                num_filters=4,
                depth=3,
                output_reduction=output_reduction,
                skip_placement="before_pool",
            )
            model.eval()

            with torch.no_grad():
                y = model(x)

            self.assertEqual(tuple(y.shape), (2, 1, expected_size, expected_size))

    def test_after_pool_unet_supports_half_and_coarser_outputs(self):
        x = torch.randn(2, 3, 64, 64)

        for output_reduction, expected_size in [(2, 32), (4, 16), (8, 8)]:
            model = UNet(
                num_filters=4,
                depth=3,
                output_reduction=output_reduction,
                skip_placement="after_pool",
            )
            model.eval()

            with torch.no_grad():
                y = model(x)

            self.assertEqual(tuple(y.shape), (2, 1, expected_size, expected_size))

    def test_after_pool_unet_rejects_full_resolution_output(self):
        with self.assertRaises(ValueError):
            UNet(
                num_filters=4,
                depth=3,
                output_reduction=1,
                skip_placement="after_pool",
            )


if __name__ == "__main__":
    unittest.main()
