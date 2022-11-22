import os
from typing import Dict, List, Optional, Tuple

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
    n_mapping: Optional[int] = None
    style_mixing_prob: float = 0.9

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
    train_steps: Optional[int]
    batch_size: int
    gen_lr: float
    gen_betas: Tuple[float, float]
    disc_lr: Optional[float]
    disc_betas: Optional[Tuple[float, float]]
    precision: int = 32
    learning_progress_path: Optional[str]
    warm_start: Optional[bool] = False
    wandb_tags: Optional[List[str]]
    test_meta_data_path: Optional[str] = None
    n_evals: Optional[int] = 10
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
        from loader import HFDataloader, MNISTDataloader, SpotifyDataloader

        if self.dataloader == "MNISTDataloader":
            return MNISTDataloader(self)
        elif self.dataloader == "SpotifyDataloader":
            return SpotifyDataloader(self)
        elif self.dataloader == "HFDatasets":
            return HFDataloader(self)
        raise ValueError(
            "dataloader has to be either MNISTDataloader, SpotifyDataloader or HFDatasets"
        )


class ProGANTrainConfig(GANTrainConfig):
    batch_size: Dict[int, int]
    n_critic: Optional[int] = 1
    gradient_penalty_weight: Optional[float] = 10.0
    fade_in_imgs: int
    burn_in_imgs: int

    @validator("fade_in_imgs", "burn_in_imgs", always=True)
    def default_fadeburn_in_imgs(cls, v, values):  # noqa: D102, N805
        """
        Increases the number of fade-in & burn-in images as PyTorch Lightning
        assigns a single step to every Generator & Discriminator update and the
        total number of updates is calculated using the desired images.
        """
        return int(v * (1 + 1 / values["n_critic"]))


StyleGANTrainConfig = ProGANTrainConfig


class DDPMTrainConfig(GANTrainConfig, extra=Extra.allow):
    # Model
    downblock_types: List[str]
    upblock_types: List[str]
    latent_size: int = None

    # Training
    lr_scheduler: str = "cosine"
    warmup_perc: float = 0.1

    overwrite_output_dir: bool = False
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_max_decay: float = 0.9999
    ddpm_beta_schedule: str = "linear"

    @validator("downblock_types", always=True)
    def down_block_types_validator(cls, v, values):  # noqa: D102, N805
        """
        Validataion of the Down Blocks definition.

        Down Blocks should be a list of "DownBlock2D" and
        "AttnDownBlock2D".
        """
        allowed_down_blocks = ("DownBlock2D", "AttnDownBlock2D")
        if not all(block in allowed_down_blocks for block in v):
            raise ValueError(
                f"down_block_types has to be a list of {allowed_down_blocks}"
            )
        return v

    @validator("upblock_types", always=True)
    def up_block_types_validator(cls, v, values):  # noqa: D102, N805
        """
        Validataion of the Up Blocks definition.

        Down Blocks should be a list of "UpBlock2D" and "AttnUpBlock2D".
        """
        allowed_up_blocks = ("UpBlock2D", "AttnUpBlock2D")
        if not all(block in allowed_up_blocks for block in v):
            raise ValueError(
                f"up_block_types has to be a list of {allowed_up_blocks}"
            )
        return v
