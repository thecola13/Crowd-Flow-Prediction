import unittest

from src.benchmark import make_experiment_grid


class BenchmarkConfigTests(unittest.TestCase):
    def test_grid_uses_shared_config_across_model_variants(self):
        configs = make_experiment_grid(
            architectures=("resnet50_ae", "vgg19_ae"),
            depths=(2, 4),
            splits=("A",),
        )

        self.assertEqual(
            [config.name for config in configs],
            [
                "resnet50_ae_depth2_split_A",
                "resnet50_ae_depth4_split_A",
                "vgg19_ae_depth2_split_A",
                "vgg19_ae_depth4_split_A",
            ],
        )
        self.assertEqual({config.data for config in configs}, {configs[0].data})
        self.assertEqual(
            {config.optimization for config in configs},
            {configs[0].optimization},
        )


if __name__ == "__main__":
    unittest.main()
