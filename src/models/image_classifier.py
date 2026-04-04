import torch
import torch.nn as nn
import lightning as pl
from lightning.pytorch.callbacks import BaseFinetuning
from timm import create_model
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


class ImageClassifier(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "resnet18.a1_in1k",
        num_classes: int = 10,
        lr: float = 5e-4,
        scheduler_factor: float = 0.2,
        scheduler_patience: int = 5,
        scheduler_monitor_value: str = "loss/val",
        scheduler_min_lr: float = 1e-6,
        scheduler_mode: str = "min",
        is_resuming: bool = False,
        show_log_progress_bar: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model_id = model_id
        self.num_classes = num_classes
        self.model = create_model(
            self.model_id,
            pretrained=True,
            num_classes=self.num_classes,
        )
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_monitor_value = scheduler_monitor_value
        self.scheduler_min_lr = scheduler_min_lr
        self.scheduler_mode = scheduler_mode
        self.is_resuming = is_resuming
        self.show_log_progress_bar = show_log_progress_bar
        metric = MetricCollection(
            {
                "accuracy": MulticlassAccuracy(num_classes=self.num_classes),
                "recall": MulticlassRecall(num_classes=self.num_classes),
                "precision": MulticlassPrecision(num_classes=self.num_classes),
                "f1": MulticlassF1Score(num_classes=self.num_classes),
            }
        )
        self.train_metric = metric.clone(prefix="train/")
        self.val_metric = metric.clone(prefix="val/")
        self.test_metric = metric.clone(prefix="test/")

    def training_step(self, batch):
        return self.shared_step(*batch, phase="train", metric=self.train_metric)

    def validation_step(self, batch):
        return self.shared_step(*batch, phase="val", metric=self.val_metric)

    def shared_step(self, x, y, phase: str, metric):
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        metric(y_hat, y)
        self.log_dict(
            metric, on_step=False, on_epoch=True, prog_bar=self.show_log_progress_bar
        )
        self.log(
            f"loss/{phase}",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=self.show_log_progress_bar,
        )
        return loss

    def test_step(self, batch):
        return self.shared_step(*batch, phase="test", metric=self.test_metric)

    def predict_step(self, batch):
        return self(batch)

    def forward(self, x):
        return torch.argmax(self.model(x), dim=1)

    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.lr, fused=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=self.scheduler_mode,
                    factor=self.scheduler_factor,
                    patience=self.scheduler_patience,
                    min_lr=self.scheduler_min_lr,
                ),
                "monitor": self.scheduler_monitor_value,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }


class ImageClassificationBaseModelFreezeUnfreezeCallback(BaseFinetuning):
    def __init__(
        self,
        unfreeze_at_epoch: int = 10,
        initial_denom_lr: int = 10,
        train_bn: bool = False,
    ):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module):
        if not pl_module.is_resuming:
            print("Freezing base model")
            self.freeze(
                list(pl_module.model.children())[:-1],
                train_bn=self.train_bn,
            )
        print("Freezing base model done")

    def finetune_function(self, pl_module, epoch, optimizer):
        if epoch == self._unfreeze_at_epoch:
            print("Unfreezing base model")
            self.unfreeze_and_add_param_group(
                modules=list(pl_module.model.children())[:-1],
                optimizer=optimizer,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=False,
            )
            print(
                f"After unfreeze_and_add_param_group, there are {len(optimizer.param_groups)} param_groups"
            )
            if len(optimizer.param_groups) == 2:
                # Should set the lr to be the one of the second param group, because we need lower lr from now on
                optimizer.param_groups[0]["lr"] = optimizer.param_groups[1]["lr"]
                print(
                    f"optimizer 0 lr is now {optimizer.param_groups[0]['lr']}, optimizer 1 lr is {optimizer.param_groups[1]['lr']}"
                )
                # Adjust the scheduler min_lrs to match the new optimizer param_groups
                scheduler = pl_module.lr_schedulers()
                if len(optimizer.param_groups) != len(scheduler.min_lrs):
                    print("Adjusting scheduler min_lrs to match optimizer param_groups")
                    scheduler.min_lrs = [scheduler.min_lrs[0]] * len(
                        optimizer.param_groups
                    )
            else:
                print("Only 1 optimizer param_group, not adjusting lr")
