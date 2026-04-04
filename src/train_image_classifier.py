import hydra
from configs.image_classifier import MainConfig


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="image_classifier_config",
)
def train(cfg: MainConfig) -> None:
    from pprint import pprint
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import torch
    from clearml import Task, Model, OutputModel
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.tuner import Tuner
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        LearningRateMonitor,
        EarlyStopping,
    )
    from lightning.pytorch.loggers import TensorBoardLogger
    from timm.data import resolve_model_data_config, create_transform

    from datamodules.image_classifier import ImageClassificationDataModule
    from models.image_classifier import (
        ImageClassifier,
        ImageClassificationBaseModelFreezeUnfreezeCallback,
    )

    pprint("Initializing Task")
    seed_everything(cfg.project_config.seed)
    Task.init(
        project_name=cfg.project_config.project_name,
        task_name=cfg.project_config.task_name,
        output_uri=cfg.project_config.output_uri,
        auto_connect_frameworks={
            "pytorch": False
        },  # we manually log the model after training
    )

    Task.current_task().connect(cfg.project_config, name="Project")
    Task.current_task().connect(cfg.module_config, name="Module")
    Task.current_task().connect(cfg.datamodule_config, name="DataModule")
    Task.current_task().connect(cfg.trainer_config, name="Trainer")
    Task.current_task().add_tags(
        [
            cfg.module_config.model_id,
            cfg.datamodule_config.dataset_path_or_id,
            f"{cfg.module_config.scheduler_mode} {cfg.module_config.scheduler_monitor_value}",
        ]
    )

    # Defining model
    pprint("Initializing model")
    # Getting checkpoint if there is
    checkpoint_path = None
    if cfg.module_config.checkpoint_model_id is not None:
        try:
            checkpoint_path = Model(
                cfg.module_config.checkpoint_model_id
            ).get_local_copy()
        except:
            pprint(
                f"Could not download checkpoint with model_id: {cfg.module_config.checkpoint_model_id}"
            )
            pprint("Exiting")
            Task.current_task().mark_failed()
            exit(-1)

        # If resuming from checkpoint, load model from checkpoint and keep the original values
        if cfg.trainer_config.resume_from_checkpoint:
            model = ImageClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                is_resuming=True,
                show_log_progress_bar=cfg.module_config.show_log_progress_bar,
            )
        # If not resuming from checkpoint, load model from checkpoint and change the values
        else:
            model = ImageClassifier.load_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_id=cfg.module_config.model_id,
                num_classes=cfg.module_config.num_classes,
                lr=cfg.module_config.lr,
                scheduler_factor=cfg.module_config.scheduler_factor,
                scheduler_patience=cfg.module_config.scheduler_patience,
                scheduler_monitor_value=cfg.module_config.scheduler_monitor_value,
                scheduler_min_lr=cfg.module_config.scheduler_min_lr,
                scheduler_mode=cfg.module_config.scheduler_mode,
                is_resuming=cfg.trainer_config.resume_from_checkpoint,
                show_log_progress_bar=cfg.module_config.show_log_progress_bar,
            )
    # If there is no checkpoint, create a new model
    else:
        model = ImageClassifier(
            model_id=cfg.module_config.model_id,
            num_classes=cfg.module_config.num_classes,
            lr=cfg.module_config.lr,
            scheduler_factor=cfg.module_config.scheduler_factor,
            scheduler_patience=cfg.module_config.scheduler_patience,
            scheduler_monitor_value=cfg.module_config.scheduler_monitor_value,
            scheduler_min_lr=cfg.module_config.scheduler_min_lr,
            scheduler_mode=cfg.module_config.scheduler_mode,
            is_resuming=False,
            show_log_progress_bar=cfg.module_config.show_log_progress_bar,
        )

    # Defining transforms
    pprint("Initializing transforms")

    data_config = resolve_model_data_config(model.model)

    train_transforms = create_transform(
        **data_config,
        is_training=True,
        auto_augment=cfg.datamodule_config.rand_augment_string,
    )
    val_transforms = create_transform(**data_config, is_training=False)

    # Defining datamodule
    pprint("Initializing datamodule")
    try:
        datamodule = ImageClassificationDataModule(
            dataset_path_or_id=cfg.datamodule_config.dataset_path_or_id,
            batch_size=cfg.datamodule_config.batch_size,
            num_workers=cfg.datamodule_config.num_workers,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
        )
    except Exception:
        pprint(
            f"Could not download dataset with dataset_id: {cfg.datamodule_config.dataset_id}"
        )
        pprint("Exiting")
        exit(-1)

    # Defining callbacks
    pprint("Setting up callbacks")
    tempdir = TemporaryDirectory().name

    checkpoint_callback = ModelCheckpoint(
        dirpath=tempdir,
        filename="image_classifier",
        monitor=cfg.module_config.scheduler_monitor_value,
        mode=cfg.module_config.scheduler_mode,
        verbose=True,
    )

    lr_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    early_stopping_callback = EarlyStopping(
        monitor=cfg.module_config.scheduler_monitor_value,
        patience=cfg.module_config.scheduler_patience * 5,
        mode=cfg.module_config.scheduler_mode,
        verbose=True,
    )

    tensorboard_callback = TensorBoardLogger(tempdir, name="ImageClassifier")

    callbacks = [
        checkpoint_callback,
        lr_monitor_callback,
        early_stopping_callback,
    ]

    if cfg.module_config.freeze_base_model:
        base_model_freeze_unfreeze_callback = (
            ImageClassificationBaseModelFreezeUnfreezeCallback(
                unfreeze_at_epoch=cfg.module_config.freeze_callback_unfreeze_at_epoch,
                initial_denom_lr=cfg.module_config.freeze_callback_initial_denom_lr,
            )
        )
        callbacks.append(base_model_freeze_unfreeze_callback)
    # Defining trainer
    pprint("Initializing trainer")
    trainer = Trainer(
        max_epochs=cfg.trainer_config.max_epochs,
        precision=cfg.trainer_config.precision,
        default_root_dir=tempdir,
        logger=tensorboard_callback,
        callbacks=callbacks,
        limit_train_batches=cfg.trainer_config.limit_train_batches,
        limit_val_batches=cfg.trainer_config.limit_val_batches,
        accumulate_grad_batches=cfg.trainer_config.accumulate_grad_batches,
        deterministic=cfg.trainer_config.deterministic,
        log_every_n_steps=1,
    )

    # Training
    if cfg.project_config.execute_remotely:
        Task.current_task().execute_remotely(
            queue_name=cfg.project_config.execution_queue
        )
    pprint("Training")
    if not cfg.trainer_config.resume_from_checkpoint:
        checkpoint_path = None
    if cfg.trainer_config.use_tuner:
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=datamodule, max_trials=10)
        tuner.lr_find(model, datamodule=datamodule, max_lr=1e-1, min_lr=1e-7)
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    pprint("Training done")

    # Saving the best model checkpoint, track in ClearML and push to S3
    pprint("Loading best checkpoint")
    model = ImageClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)
    tags = [
        f"model {cfg.module_config.model_id}",
        f"dataset_id {cfg.datamodule_config.dataset_path_or_id}",
    ]
    # Saving only the torch module
    torch_model_path = Path(tempdir) / "model.pth"
    torch.save(model.model.state_dict(), f=torch_model_path)
    pprint(f"Saving model with tags: {tags}")

    # Add metadata to the model to facilitate loading for inference after
    metadata = {
        "model_name": cfg.module_config.model_id,
        "num_classes": cfg.module_config.num_classes,
        "in_chans": 3,
    }
    pprint("Saving the torch model to ClearML with class labels and metadata")
    output_model = OutputModel(
        task=Task.current_task(),
        name="ImageClassifier",
        tags=tags,
        framework="torch",
    )
    output_model.update_weights(
        weights_filename=str(torch_model_path),
        auto_delete_file=False,
        upload_uri=cfg.project_config.output_uri,
    )
    output_model.update_labels(datamodule.class_labels)
    for k, v in metadata.items():
        output_model.set_metadata(k, v, v_type=type(v).__name__)
    pprint("Pushing model to S3")
    OutputModel.wait_for_uploads()

    pprint("Evaluating")
    try:
        test_results = trainer.test(model=model, datamodule=datamodule)
        pprint(test_results)
        task_logger = Task.current_task().get_logger()
        for key, value in test_results[0].items():
            task_logger.report_single_value(f"{key}", value)
            output_model.report_single_value(f"{key}", value)
    except Exception as e:
        pprint("Could not evaluate model")
        print(e)
    pprint("Done")


if __name__ == "__main__":
    train()
