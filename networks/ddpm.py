import os
from typing import Dict

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch import nn
from tqdm import tqdm

from config import DDPMTrainConfig


class DDPM(nn.Module):
    def __init__(self, config: DDPMTrainConfig, accelerator: Accelerator):
        """
        Denoising Diffusion Probabilistic Model.

        Initializes a UNet2D model and can be used to train a Denoising Diffusion Probabilistic Model in order to generate images by running the train method

        Args:
            config: Config object containing the model and training parameters
            accelerator: Diffusors Accelerator to train on multiple devices

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
        self.model = accelerator.prepare(self.model)
        self.ema_model = EMAModel(
            self.model,
            inv_gamma=self.config.ema_inv_gamma,
            power=self.config.ema_power,
            max_value=self.config.ema_max_decay,
        )

        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None

        self.global_step = 0

    def train(self, trainset, accelerator: Accelerator):
        """
        Trains the model to generate images.

        Args:
            trainset: Torch Dataset
            accelerator: Diffusors Accelerator to train on multiple devices
        """
        self.model.train()
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

        self.optimizer = accelerator.prepare(self.optimizer)
        trainset = accelerator.prepare(trainset)
        self.lr_scheduler = accelerator.prepare(self.lr_scheduler)

        if self.config.output_dir is not None:  # TODO: Fix outputdir
            os.makedirs(self.config.output_dir, exist_ok=True)

        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

        progress_bar = tqdm(
            total=self.config.train_steps,
            disable=not accelerator.is_local_main_process,
        )
        for _step, batch in enumerate(trainset):
            loss = self.train_step(batch, accelerator=accelerator)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                    "step": self.global_step,
                }
                logs["ema_decay"] = self.ema_model.decay
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=self.global_step)
        progress_bar.close()

    def train_step(
        self, batch: Dict, accelerator: Accelerator
    ) -> torch.Tensor:
        """
        Performs a single gradient update on a given batch.

        Args:
            batch: Datasample dictionary. Images should be at key "input"
            accelerator: Diffusors Accelerator to train on multiple devices
        """
        clean_images = batch["input"]

        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (self.config.batch_size,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep. This is the forward diffusion process
        noisy_images = self.noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )
        with accelerator.accumulate(self.model):
            # Predict the noise residual
            model_output = self.model(noisy_images, timesteps).sample
            loss = F.mse_loss(model_output, noise)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.ema_model.step(self.model)
        self.optimizer.zero_grad()
        return loss
