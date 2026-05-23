import argparse

from src.benchmark import (
    DataConfig,
    LoggingConfig,
    OptimizationConfig,
    TrainerConfig,
    make_experiment_grid,
    run_benchmark,
)


def parse_csv(value: str, cast=str):
    if value.lower() == "all":
        return None
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified crowd-count benchmark.")
    parser.add_argument("--architectures", default="resnet50_ae,vgg19_ae")
    parser.add_argument("--depths", default="2,3,4")
    parser.add_argument("--splits", default="A,B")
    parser.add_argument("--data-folder", default="./data/ShanghaiTech")
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
    splits = parse_csv(args.splits) or ("A", "B")

    configs = make_experiment_grid(
        architectures=architectures,
        depths=depths,
        splits=splits,
        data=DataConfig(
            data_folder=args.data_folder,
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

    results = run_benchmark(configs)
    failed = [result for result in results if result["status"] != "ok"]
    if failed:
        for result in failed:
            print(f"{result['name']} failed: {result['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
