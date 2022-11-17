import os
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DModel, __version__
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DDPMPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate
from packaging import version
from torch import nn
from tqdm import tqdm

from config import DDPMTrainConfig

diffusers_version = version.parse(version.parse(__version__).base_version)


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
        evaluate_every_n_steps = self.config.train_steps // self.config.n_evals
        for _step, batch in enumerate(trainset):
            logs = self.train_step(batch, accelerator=accelerator)
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=self.global_step)
            if self.global_step % evaluate_every_n_steps == 0:
                # This needs to be executed before we save a
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(
                            self.ema_model.averaged_model
                        ),
                        scheduler=self.noise_scheduler,
                    )
                    images_processed = self.evaluate(pipeline)
                    # Log images
                    accelerator.trackers[0].writer.add_images(
                        "test_samples",
                        images_processed.transpose(0, 3, 1, 2),
                        self.global_step,
                    )
                pipeline.save_pretrained(self.config.output_dir)
            progress_bar.update(1)
            self.global_step += 1
        accelerator.wait_for_everyone()
        accelerator.end_training()
        progress_bar.close()

    def evaluate(self, pipeline) -> np.ndarray:
        """
        Performs a single evaluation step at the current state of the model

        Args:
            pipeline: diffusers DDPMPipeline to perfrom inference
        """
        deprecate(
            "todo: remove this check",
            "0.10.0",
            "when the most used version is >= 0.8.0",
        )
        if diffusers_version < version.parse("0.8.0"):
            generator = torch.manual_seed(0)
        else:
            generator = torch.Generator(device=pipeline.device).manual_seed(0)
            # run pipeline in inference (sample random noise and denoise)
        images = pipeline(
            generator=generator,
            batch_size=self.config.batch_size,
            output_type="numpy",
        ).images

        # denormalize the images
        images_processed = (images * 255).round().astype("uint8")
        return images_processed

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

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            logs = {
                "loss": loss.detach().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
                "step": self.global_step,
                "ema_decay": self.ema_model.decay,
            }
        return logs
