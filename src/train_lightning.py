import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from pytorch_lightning.utilities.types import OptimizerLRSchedulerConfig

from src.metrics import (
    count_mae,
    count_mse,
    count_rmse,
    pixelwise_mae,
    pixelwise_rmse,
)
from src.models import get_model
from src.utils import plot_dec_steps_batch, resize_density_map_count_preserving


class LitDensityEstimator(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "resnet50",
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        optimizer_name: str = "adamw",
        scheduler_name: str = "one_cycle",
        pretrained: bool = True,
        freeze_encoder: bool = False,
        device: torch.device | None = None,
        log_images: bool = True,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.log_images = log_images
        self.model = get_model(
            model_name,
            pretrained=pretrained,
            freeze_encoder=freeze_encoder,
            **model_kwargs
        )
        self.criterion = nn.MSELoss()
        self.model.to(device)

    def forward(self, x, return_intermediates: bool = False):
        return self.model(x, return_intermediates=return_intermediates)

    @staticmethod
    def _final_density(output):
        if isinstance(output, (list, tuple)):
            return output[-1]
        return output

    @staticmethod
    def _match_target_size(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape[-2:] == target.shape[-2:]:
            return pred
        return resize_density_map_count_preserving(pred, size=target.shape[-2:])

    def training_step(self, batch, batch_idx):
        img, gt = batch
        pred = self._match_target_size(self._final_density(self(img)), gt)
        loss = self.criterion(pred, gt)
        mae = pixelwise_mae(pred, gt)
        rmse = pixelwise_rmse(pred, gt)

        self.log('train/mse', loss, on_epoch=True, prog_bar=True)
        self.log('train/mae', mae, on_epoch=True)
        self.log('train/rmse', rmse, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, name="val"):
        img, gt = batch
        preds = self(img, return_intermediates=True)
        pred = self._match_target_size(self._final_density(preds), gt)
        loss = self.criterion(pred, gt)
        mae = pixelwise_mae(pred, gt)
        rmse = pixelwise_rmse(pred, gt)

        # count metrics
        count_mae_val = count_mae(pred, gt)
        count_mse_val = count_mse(pred, gt)
        count_rmse_val = count_rmse(pred, gt)

        self.log(f'{name}/mse', loss, on_epoch=True, prog_bar=True)
        self.log(f'{name}/mae', mae, on_epoch=True)
        self.log(f'{name}/rmse', rmse, on_epoch=True)

        # Log count metrics
        self.log(f'{name}/count_mae', count_mae_val, on_epoch=True)
        self.log(f'{name}/count_mse', count_mse_val, on_epoch=True)
        self.log(f'{name}/count_rmse', count_rmse_val, on_epoch=True)

        # Log images to W&B for the first batch only
        if self.log_images and batch_idx == 0 and isinstance(self.logger, pl_loggers.WandbLogger):
            import matplotlib.pyplot as plt
            import wandb

            if not isinstance(preds, (list, tuple)):
                preds = [pred]
            fig = plot_dec_steps_batch(img, gt, preds)
            # Log figure
            self.logger.experiment.log({
                f"{name}_batch_images_epoch_{self.current_epoch}": wandb.Image(fig)
            })
            plt.close(fig)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        if self.optimizer_name != "adamw":
            raise ValueError(f"Unsupported optimizer '{self.optimizer_name}'")

        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
        if self.scheduler_name == "one_cycle":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=total_steps,
                pct_start=0.3,
                anneal_strategy='cos',
                final_div_factor=1e4
            )
            interval = "step"
        elif self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=max(self.trainer.max_epochs, 1))
            interval = "epoch"
        elif self.scheduler_name in {"none", None}:
            return optimizer
        else:
            raise ValueError(f"Unsupported scheduler '{self.scheduler_name}'")

        # Return compatible config for PyTorch Lightning
        return OptimizerLRSchedulerConfig({ 
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': interval,
                'frequency': 1
            }
        })
