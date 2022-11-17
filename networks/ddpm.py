import math
import os

import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch import nn

from config import DDPMTrainConfig


class DDPM(nn.Module):
    def __init__(self, config: DDPMTrainConfig) -> None:
        """
        Denoising Diffusion Probabilistic Model.

        Initializes a UNet2D model and can be used to train a Denoising Diffusion Probabilistic Model in order to generate images by running the train method

        Args:
            config: Config object containing the model and training parameters

        Reference:
            https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf
        """
        super(DDPM, self).__init__()
        self.config = config
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.channels,
            out_channels=config.channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=config.downblock_types,
            up_block_types=config.upblock_types,
        )
        self.ema_model = EMAModel(
            self.model,
            inv_gamma=self.config.ema_inv_gamma,
            power=self.config.ema_power,
            max_value=self.config.ema_max_decay,
        )

        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None

    def train(self, trainset, accelerator):
        """
        Trains the model to generate images.

        Args:
            trainset: Torch Dataset
            accelerator: Diffusors Accelerator to train on multiple devices
        """
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.config.train_steps,
            beta_schedule=self.config.ddpm_beta_schedule,
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.gen_lr,
            betas=self.config.gen_betas,
            weight_decay=1e-6,
        )
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.warmup_perc,
            num_training_steps=self.config.train_steps
            // self.config.gradient_accumulation_steps,
        )

        self.model = accelerator.prepare(self.model)
        self.optimizer = accelerator.prepare(self.optimizer)
        self.trainset = accelerator.prepare(self.trainset)
        self.lr_scheduler = accelerator.prepare(self.lr_scheduler)

        num_steps_per_epoch = math.ceil(
            len(trainset) / self.config.gradient_accumulation_steps
        )

        if self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)

        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)
