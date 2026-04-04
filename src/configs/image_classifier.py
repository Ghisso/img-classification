from dataclasses import dataclass, field
from typing import Optional, Union

from hydra.core.config_store import ConfigStore


@dataclass
class ProjectConfig:
    project_name: str = "ImageClassifier"
    task_name: str = "training"
    seed: int = 2024
    output_uri: Union[bool, str] = False
    execute_remotely: bool = False
    execution_queue: str = "default"


@dataclass
class ModuleConfig:
    model_id: str = "resnet18.a1_in1k"
    lr: float = 1e-3
    num_classes: int = 9
    scheduler_factor: float = 0.2
    scheduler_patience: int = 5
    scheduler_monitor_value: str = "loss/val_epoch"
    scheduler_min_lr: float = 1e-8
    scheduler_mode: str = "min"
    freeze_base_model: bool = True
    freeze_callback_unfreeze_at_epoch: int = 20
    freeze_callback_initial_denom_lr: int = 100
    checkpoint_model_id: Optional[str] = None
    show_log_progress_bar: bool = True


@dataclass
class DataModuleConfig:
    dataset_path_or_id: str = "data/splits"
    batch_size: int = 512
    num_workers: int = 0
    rand_augment_string: str = (
        "rand-m5-n2-mstd0.5-inc1"  # Check https://timm.fast.ai/RandAugment to understand how it works
    )


@dataclass
class TrainerConfig:
    max_epochs: int = 200
    precision: str = "16-true"
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None
    accumulate_grad_batches: int = 1
    deterministic: bool = True
    resume_from_checkpoint: bool = False
    use_tuner: bool = False


@dataclass
class MainConfig:
    project_config: ProjectConfig = field(default_factory=ProjectConfig)
    module_config: ModuleConfig = field(default_factory=ModuleConfig)
    datamodule_config: DataModuleConfig = field(default_factory=DataModuleConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)


cs = ConfigStore.instance()
cs.store(
    name="main_config",
    node=MainConfig,
)

cs.store(
    name="main_config",
    node=MainConfig,
)
