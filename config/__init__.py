import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
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
    add_release_year: bool = False
    n_blocks: Optional[int] = None

    # Tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = "duck-and-cover"
    wandb_run_id: Optional[str] = None

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

    @validator("n_blocks", always=True)
    def default_n_blocks(cls, v, values):  # noqa: D102, N805
        return int(np.log2(values["image_size"]) - 1)


class GANTrainConfig(GANConfig):
    # Data
    dataloader: str
    meta_data_path: Optional[str]

    # Training:
    batch_size: Union[int, List[int]]
    gen_lr: float
    gen_betas: Tuple[float, float]
    disc_lr: float
    disc_betas: Tuple[float, float]
    precision: int = 32
    learning_progress_path: Optional[str]
    warm_start: Optional[bool] = False
    wandb_tags: Optional[List[str]]
    test_meta_data_path: Optional[str] = None
    n_critic: Optional[int] = 1
    gradient_penalty_weight: Optional[float] = 10.0
    train_steps: int
    eval_rate: Optional[int]
    ema_beta: float = 0.999

    @validator("meta_data_path", always=True)
    def meta_data_path_validator(cls, v, values):  # noqa: D102, N805
        if values["dataloader"] == "SpotifyDataloader" and not v:
            raise ValueError(
                "meta_data_path has to be set for SpotifyDataloader"
            )
        return v

    @validator("learning_progress_path", always=True)
    def default_learning_progress_path(cls, v, values):  # noqa: D102, N805
        return v or os.path.join(
            "learning_progress", values["unique_experiment_name"]
        )

    @validator("eval_rate", always=True)
    def default_eval_rate(cls, v, values):  # noqa: D102, N805
        return v or values["train_steps"] // 32

    @validator("train_steps", always=True)
    def default_train_steps(cls, v, values):  # noqa: D102, N805
        """
        Increments the step size as discriminator step and generator step are
        seen as one step each.

        Args:
            v: Number of steps.
            values: Values of the config.
        """
        return int(v * (1 + 1 / values["n_critic"]))

    @validator("wandb_tags", always=True)
    def default_wandb_tags(cls, v, values):  # noqa: D102, N805
        tags = []
        if values["dataloader"] == "MNISTDataloader":
            tags.append("mnist")
        else:
            tags.append("spotify")
        if values["add_release_year"]:
            tags.append("release_year")
        return v or tags

    def get_dataloader(self):
        """
        Return the dataloader class based on the dataloader definition.
        """
        from loader import MNISTDataloader, SpotifyDataloader

        if self.dataloader == "MNISTDataloader":
            return MNISTDataloader(self)
        elif self.dataloader == "SpotifyDataloader":
            return SpotifyDataloader(self)
        raise ValueError(
            "dataloader has to be either MNISTDataloader or DataLoader"
        )
