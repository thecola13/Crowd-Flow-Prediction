import argparse

from src.benchmark import (
    DataConfig,
    LoggingConfig,
    OptimizationConfig,
    TrainerConfig,
    make_experiment_grid,
    make_output_resolution_ablation,
    make_receptive_field_ablation,
    make_skip_placement_ablation,
    run_benchmark,
)


def parse_csv(value: str, cast=str):
    if value.lower() == "all":
        return None
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified crowd-count benchmark.")
    parser.add_argument(
        "--ablation",
        choices=("grid", "receptive_field", "output_resolution", "skip_placement"),
        default="grid",
    )
    parser.add_argument("--architectures", default="resnet50_ae,vgg19_ae")
    parser.add_argument("--depths", default="2,3,4")
    parser.add_argument("--output-reductions", default="2")
    parser.add_argument("--skip-placements", default="after_pool")
    parser.add_argument("--splits", default="A,B")
    parser.add_argument("--data-folder", default="./data/ShanghaiTech")
    parser.add_argument("--dataset", default="sha")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--eval-data-folder", default=None)
    parser.add_argument("--eval-dataset", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="one_cycle")
    parser.add_argument("--project", default="crowd-flow-benchmark")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-image-logs", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    architectures = parse_csv(args.architectures) or ("resnet50_ae", "vgg19_ae")
    depths = parse_csv(args.depths, int) or (2, 3, 4)
    output_reductions = parse_csv(args.output_reductions, int) or (2,)
    skip_placements = parse_csv(args.skip_placements) or ("after_pool",)
    splits = parse_csv(args.splits) or ("A", "B")

    shared = dict(
        data=DataConfig(
            data_folder=args.data_folder,
            dataset_name=args.dataset,
            train_split=args.train_split,
            test_split=args.test_split,
            eval_data_folder=args.eval_data_folder,
            eval_dataset_name=args.eval_dataset,
            eval_split=args.eval_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        optimization=OptimizationConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
        ),
        trainer=TrainerConfig(
            max_epochs=args.max_epochs,
            fast_dev_run=args.fast_dev_run,
        ),
        logging=LoggingConfig(
            use_wandb=not args.no_wandb,
            project=args.project,
            log_images=not args.no_image_logs,
        ),
    )

    if args.ablation == "receptive_field":
        configs = make_receptive_field_ablation(
            architecture=architectures[0],
            depths=depths,
            output_reduction=output_reductions[0],
            split=splits[0],
            **shared,
        )
    elif args.ablation == "output_resolution":
        configs = make_output_resolution_ablation(
            architecture=architectures[0],
            depth=depths[0],
            output_reductions=output_reductions,
            split=splits[0],
            **shared,
        )
    elif args.ablation == "skip_placement":
        configs = make_skip_placement_ablation(
            architecture=architectures[0],
            depth=depths[0],
            output_reduction=output_reductions[0],
            split=splits[0],
            **shared,
        )
    else:
        configs = make_experiment_grid(
            architectures=architectures,
            depths=depths,
            output_reductions=output_reductions,
            skip_placements=skip_placements,
            splits=splits,
            **shared,
        )

    results = run_benchmark(configs)
    failed = [result for result in results if result["status"] != "ok"]
    if failed:
        for result in failed:
            print(f"{result['name']} failed: {result['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
