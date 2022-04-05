import os
from typing import Dict, List, Optional, Union

import pytorch_lightning as pl
import randomname
import yaml
from pydantic import BaseModel, Extra, validator


class GANConfig(BaseModel, extra=Extra.forbid):
    # Experiment
    experiment_name: str
    unique_experiment_name: Optional[str]

    # Model
    image_size: Optional[int] = 256
    channels: int = 3
    latent_size: int

    # Tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = "duck_and_cover"
    wandb_run_id: Optional[str] = None
    wandb_tags: Optional[List] = []

    def __init__(self, config_path: str = None, config_dict: Dict = None):
        """
        Application class that can be used to orchestrate the training and
        evaluation of a QQBertModel.

        Args:
            config_path: Path to a config file.
            config_dict: Dictionary of arguments for the config. Ignored if path
                to the config file is given.
        """
        if config_path:
            with open(config_path, "r") as stream:
                config_dict = yaml.safe_load(stream)
        super().__init__(**config_dict)

    @validator("unique_experiment_name", always=True)
    def default_unique_experiment_name(cls, v, values):  # noqa: D102, N805
        return v or values["experiment_name"] + "_" + randomname.get_name()


class GANTrainConfig(GANConfig):
    # Data
    dataloader: str

    # Training:
    train_steps: int
    batch_size: Union[int, List[int]]
    minibatch_size: int = 4
    callbacks: Optional[List[Dict]]
    lr: Optional[float] = 1e-4
    precision: int = 32
    learning_progress_path: Optional[str]
    warm_start: Optional[bool] = False

    @validator("learning_progress_path", always=True)
    def default_unique_experiment_name(cls, v, values):  # noqa: D102, N805
        return v or os.path.join(
            "learning_progress", values["unique_experiment_name"]
        )

    def get_callbacks(self) -> List[pl.callbacks.Callback]:  # noqa: WPS615
        """
        Returns a list of Pytorch Lightning callbacks based on the config
        given.
        """
        cbs = []
        for cb in self.callbacks:
            class_path = cb.get("class_path")
            class_name = class_path.split(".")[-1]
            init_args = cb.get("init_args")
            cb_class = getattr(pl.callbacks, class_name)
            cbs.append(cb_class(**init_args))
        return cbs

    def get_wandb_logger(self) -> pl.loggers.WandbLogger:  # noqa: WPS615
        """
        Returns a Pytorch Lightning W&B Logger based on the config given.
        """
        if self.use_wandb and self.wandb_project:
            return pl.loggers.WandbLogger(
                project=self.wandb_project, name=self.unique_experiment_name
            )

    def get_dataloader(self):
        """
        Return the dataloader class based on the dataloader definition.
        """
        from loader import DataLoader, MNISTDataloader

        if self.dataloader == "MNISTDataloader":
            return MNISTDataloader(self)
        elif self.dataloader == "DataLoader":
            return DataLoader(self)
        raise ValueError(
            "dataloader has to be either MNISTDataloader or DataLoader"
        )

    def get_loss(self):  # noqa: WPS615
        """
        Returns the loss function based on the loss definition given in the
        config.
        """
        from dense_passage_retrieval.modules.losses import (
            BatchHardTripletLoss,
            MultipleNegativesRankingLoss,
        )

        if self.loss == "BatchHardTripletLoss":
            return BatchHardTripletLoss(
                similarity_function=self.similarity, margin=self.margin
            )
        elif self.loss == "MultipleNegativesRankingLoss":
            return MultipleNegativesRankingLoss(
                similarity_function=self.similarity
            )
        raise ValueError(
            "loss has to be either BatchHardTripletLoss or MultipleNegativesRankingLoss"
        )


class GANEvalConfig(GANConfig, extra=Extra.ignore):
    # Model
    huggingface_model: bool = False
    pooling: str = "cls"
    model_weights: Optional[str]

    # Evaluation
    public_ds: Optional[List[str]]
