from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class DataConfig:
    data_folder: str = "./data/ShanghaiTech"
    dataset_name: str = "sha"
    train_split: str = "train"
    test_split: str = "test"
    eval_data_folder: str | None = None
    eval_dataset_name: str | None = None
    eval_split: str | None = None
    validation_split: float = 0.1
    seed: int = 42
    sigma: float = 5.0
    batch_size: int = 8
    num_workers: int = 4
    input_size: tuple[int, int] = (384, 384)
    density_map_size: tuple[int, int] | None = None
    use_crop_augmentation: bool = False
    crop_size: tuple[int, int] = (256, 256)
    crops_per_image: int = 1
    full_image_probability: float = 0.5
    scale_jitter: tuple[float, float] = (0.75, 1.25)
    horizontal_flip_probability: float = 0.5
    photometric_jitter: float = 0.15
    max_crop_resample_attempts: int = 30


@dataclass(frozen=True)
class OptimizationConfig:
    lr: float = 5e-4
    weight_decay: float = 0.0
    optimizer_name: str = "adamw"
    scheduler_name: str = "one_cycle"


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int = 200
    log_every_n_steps: int = 10
    patience: int = 10
    min_delta: float = 1e-5
    devices: int = 1
    default_root_dir: str = "./outputs"
    checkpoint_root: str = "./models/checkpoints"
    monitor: str = "val/mse"
    monitor_mode: str = "min"
    fast_dev_run: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    use_wandb: bool = True
    project: str = "crowd-flow-benchmark"
    log_images: bool = True


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    depth: int = 4
    output_reduction: int = 2
    skip_placement: str = "after_pool"
    label: str | None = None
    pretrained: bool = True
    freeze_encoder: bool = False
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return (
            self.label
            or f"{self.architecture}_depth{self.depth}"
            f"_out{self.output_reduction}_skip-{self.skip_placement}"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelSpec
    split: str
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def name(self) -> str:
        eval_name = self.data.eval_dataset_name or self.data.dataset_name
        dataset_label = self.data.dataset_name
        if eval_name != dataset_label:
            dataset_label = f"{dataset_label}_to_{eval_name}"
        return f"{self.model.name}_{dataset_label}_split_{self.split}"


def make_experiment_grid(
    architectures: Iterable[str] = ("resnet50_ae", "vgg19_ae"),
    depths: Iterable[int] = (2, 3, 4),
    output_reductions: Iterable[int] = (2,),
    skip_placements: Iterable[str] = ("after_pool",),
    splits: Iterable[str] = ("A", "B"),
    data: DataConfig | None = None,
    optimization: OptimizationConfig | None = None,
    trainer: TrainerConfig | None = None,
    logging: LoggingConfig | None = None,
) -> list[ExperimentConfig]:
    shared_data = data or DataConfig()
    shared_optimization = optimization or OptimizationConfig()
    shared_trainer = trainer or TrainerConfig()
    shared_logging = logging or LoggingConfig()

    return [
        ExperimentConfig(
            model=ModelSpec(
                architecture=architecture,
                depth=depth,
                output_reduction=output_reduction,
                skip_placement=skip_placement,
            ),
            split=split,
            data=shared_data,
            optimization=shared_optimization,
            trainer=shared_trainer,
            logging=shared_logging,
        )
        for split in splits
        for architecture in architectures
        for depth in depths
        for output_reduction in output_reductions
        for skip_placement in skip_placements
    ]


def make_receptive_field_ablation(
    architecture: str = "resnet50_ae",
    depths: Iterable[int] = (2, 3, 4),
    output_reduction: int = 2,
    split: str = "A",
    **shared,
) -> list[ExperimentConfig]:
    return make_experiment_grid(
        architectures=(architecture,),
        depths=depths,
        output_reductions=(output_reduction,),
        skip_placements=("after_pool",),
        splits=(split,),
        **shared,
    )


def make_output_resolution_ablation(
    architecture: str = "vgg19_ae",
    depth: int = 4,
    output_reductions: Iterable[int] = (1, 2, 4),
    split: str = "A",
    **shared,
) -> list[ExperimentConfig]:
    return make_experiment_grid(
        architectures=(architecture,),
        depths=(depth,),
        output_reductions=output_reductions,
        skip_placements=("after_pool",),
        splits=(split,),
        **shared,
    )


def make_skip_placement_ablation(
    architecture: str = "unet",
    depth: int = 4,
    output_reduction: int = 2,
    split: str = "A",
    **shared,
) -> list[ExperimentConfig]:
    return make_experiment_grid(
        architectures=(architecture,),
        depths=(depth,),
        output_reductions=(output_reduction,),
        skip_placements=("before_pool", "after_pool"),
        splits=(split,),
        **shared,
    )


def _make_data_module(config: ExperimentConfig, device):
    from src.data_loader import CropAugmentationConfig, CrowdCountingDataModule

    data = config.data
    density_map_size = data.density_map_size
    if density_map_size is None:
        reduction = config.model.output_reduction
        density_map_size = tuple(dim // reduction for dim in data.input_size)

    return CrowdCountingDataModule(
        data_folder=data.data_folder,
        dataset_name=data.dataset_name,
        train_split=data.train_split,
        test_split=data.test_split,
        eval_data_folder=data.eval_data_folder,
        eval_dataset_name=data.eval_dataset_name,
        eval_split=data.eval_split,
        validation_split=data.validation_split,
        seed=data.seed,
        sigma=data.sigma,
        return_count=False,
        batch_size=data.batch_size,
        num_workers=data.num_workers,
        input_size=data.input_size,
        density_map_size=density_map_size,
        device=device,
        crop_augmentation=CropAugmentationConfig(
            enabled=data.use_crop_augmentation,
            crop_size=data.crop_size,
            crops_per_image=data.crops_per_image,
            full_image_probability=data.full_image_probability,
            scale_jitter=data.scale_jitter,
            horizontal_flip_probability=data.horizontal_flip_probability,
            photometric_jitter=data.photometric_jitter,
            max_crop_resample_attempts=data.max_crop_resample_attempts,
        ),
    )


def _make_logger(config: ExperimentConfig):
    if not config.logging.use_wandb or config.trainer.fast_dev_run:
        return False

    from pytorch_lightning.loggers import WandbLogger

    return WandbLogger(
        project=config.logging.project,
        name=config.name,
        tags=[
            f"model_{config.model.architecture}",
            f"depth_{config.model.depth}",
            f"out_{config.model.output_reduction}",
            f"skip_{config.model.skip_placement}",
            f"train_{config.data.dataset_name}",
            f"eval_{config.data.eval_dataset_name or config.data.dataset_name}",
            f"split_{config.split}",
        ],
    )


def _make_callbacks(config: ExperimentConfig, has_logger: bool) -> list:
    from pytorch_lightning.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    checkpoint_dir = os.path.join(config.trainer.checkpoint_root, config.name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=config.name + "_{epoch:02d}",
        save_top_k=1,
        monitor=config.trainer.monitor,
        mode=config.trainer.monitor_mode,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.trainer.monitor,
        patience=config.trainer.patience,
        mode=config.trainer.monitor_mode,
        verbose=True,
        min_delta=config.trainer.min_delta,
    )
    callbacks = [checkpoint_callback, early_stop_callback]
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def run_experiment(config: ExperimentConfig, device=None) -> dict:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger

    from src.train_lightning import LitDensityEstimator
    from src.utils import compute_receptive_field, get_device

    pl.seed_everything(config.data.seed)
    device = device or get_device()

    data_module = _make_data_module(config, device)
    model = LitDensityEstimator(
        model_name=config.model.architecture,
        depth=config.model.depth,
        output_reduction=config.model.output_reduction,
        skip_placement=config.model.skip_placement,
        lr=config.optimization.lr,
        weight_decay=config.optimization.weight_decay,
        optimizer_name=config.optimization.optimizer_name,
        scheduler_name=config.optimization.scheduler_name,
        pretrained=config.model.pretrained,
        freeze_encoder=config.model.freeze_encoder,
        device=device,
        log_images=config.logging.log_images,
        **config.model.model_kwargs,
    )

    logger = _make_logger(config)
    if isinstance(logger, WandbLogger):
        logger.experiment.config.update(asdict(config))
        logger.experiment.config.update({"receptive_field": compute_receptive_field(model)})

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        default_root_dir=config.trainer.default_root_dir,
        logger=logger,
        callbacks=_make_callbacks(config, has_logger=logger is not False),
        accelerator=device.type,
        devices=config.trainer.devices,
        fast_dev_run=config.trainer.fast_dev_run,
    )

    try:
        trainer.fit(model, datamodule=data_module)
        test_results = trainer.test(model, datamodule=data_module)
        checkpoint_dir = os.path.join(config.trainer.checkpoint_root, config.name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.save_checkpoint(os.path.join(checkpoint_dir, config.name + ".ckpt"))
        return {"name": config.name, "status": "ok", "test": test_results}
    finally:
        if isinstance(logger, WandbLogger):
            import wandb

            wandb.finish()


def run_benchmark(configs: Iterable[ExperimentConfig]) -> list[dict]:
    from src.utils import get_device

    device = get_device()
    results = []
    for config in configs:
        try:
            results.append(run_experiment(config, device=device))
        except Exception as exc:
            results.append({"name": config.name, "status": "failed", "error": str(exc)})
            if config.logging.use_wandb:
                import wandb

                wandb.finish(quiet=True)
    return results
