import unittest

from src.benchmark import (
    make_experiment_grid,
    make_output_resolution_ablation,
    make_receptive_field_ablation,
    make_skip_placement_ablation,
)


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
                "resnet50_ae_depth2_out2_skip-after_pool_split_A",
                "resnet50_ae_depth4_out2_skip-after_pool_split_A",
                "vgg19_ae_depth2_out2_skip-after_pool_split_A",
                "vgg19_ae_depth4_out2_skip-after_pool_split_A",
            ],
        )
        self.assertEqual({config.data for config in configs}, {configs[0].data})
        self.assertEqual(
            {config.optimization for config in configs},
            {configs[0].optimization},
        )

    def test_receptive_field_ablation_varies_only_depth(self):
        configs = make_receptive_field_ablation(
            architecture="resnet50_ae",
            depths=(2, 3, 4),
            output_reduction=2,
            split="A",
        )

        self.assertEqual([config.model.depth for config in configs], [2, 3, 4])
        self.assertEqual({config.model.architecture for config in configs}, {"resnet50_ae"})
        self.assertEqual({config.model.output_reduction for config in configs}, {2})
        self.assertEqual({config.model.skip_placement for config in configs}, {"after_pool"})

    def test_output_resolution_ablation_varies_only_output_reduction(self):
        configs = make_output_resolution_ablation(
            architecture="vgg19_ae",
            depth=4,
            output_reductions=(1, 2, 4),
            split="A",
        )

        self.assertEqual([config.model.output_reduction for config in configs], [1, 2, 4])
        self.assertEqual({config.model.architecture for config in configs}, {"vgg19_ae"})
        self.assertEqual({config.model.depth for config in configs}, {4})

    def test_skip_placement_ablation_varies_only_skip_placement(self):
        configs = make_skip_placement_ablation(
            architecture="unet",
            depth=4,
            output_reduction=2,
            split="A",
        )

        self.assertEqual(
            [config.model.skip_placement for config in configs],
            ["before_pool", "after_pool"],
        )
        self.assertEqual({config.model.architecture for config in configs}, {"unet"})
        self.assertEqual({config.model.depth for config in configs}, {4})
        self.assertEqual({config.model.output_reduction for config in configs}, {2})


if __name__ == "__main__":
    unittest.main()
