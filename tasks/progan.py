from typing import Dict, List

import numpy as np
import torch
from torch import nn

from tasks.wgan import WGANTask
from utils import logger


class ProGANTask(WGANTask):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        release_year_scaler=None,
        block: int = 0,
        n_critic: int = 1,
    ):
        super(ProGANTask, self).__init__(
            generator=generator, discriminator=discriminator, n_critic=n_critic
        )
        self.release_year_scaler = release_year_scaler
        self.block = block
        self.burn_in_images_shown = 0
        self.fade_in_images_shown = 0
        self.alpha = 1
        self.automatic_optimization = False
        self._alphas = None

    def get_phase(self):
        batch_size = (
            self.trainer._data_connector._train_dataloader_source.dataloader().batch_size
        )
        fade_images_shown = self.fade_in_images_shown
        if (
            self.block == 0
            or (fade_images_shown // batch_size) == self.trainer.steps
        ):
            return "burn_in"
        return "fade_in"

    def on_fit_start(self):
        batch_size = (
            self.trainer._data_connector._train_dataloader_source.dataloader().batch_size
        )
        self._alphas = np.linspace(0, 1, self.trainer.global_step).tolist()
        img_resolution = 2 ** (self.block + 2)
        if self.get_phase() == "fade_in":
            logger.info(f"Phase: Fade in for resolution {img_resolution}")
            for _ in range(self.fade_in_images_shown // batch_size):
                self._alphas.pop(0)
            self.alpha = self._alphas.pop(0)
        else:
            logger.info(f"Phase: Burn in for resolution {img_resolution}")

    def training_step(self, batch):
        """
        Trains the ProGAN on a single batch of data. Obtains the phase (`burn-
        in` or `phade-in`) and trains the GAN accordingly.

        Args:
            batch: Batch to train on
        """
        phase = self.get_phase()
        if phase == "fade_in":
            self.alpha = self._alphas.pop(0)
        self.train_on_batch(batch)
        if phase == "fade_in":
            self.fade_in_images_shown += len(batch.get("images"))
        elif phase == "burn_in":
            self.burn_in_images_shown += len(batch.get("images"))
        self.log(
            "train/images_shown",
            self.fade_in_images_shown + self.burn_in_images_shown,
        )

    def train_discriminator(
        self, batch: Dict[str, torch.Tensor], noise: torch.Tensor
    ):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Real input images used for training
            noise: Noise to use from image generation

        Returns:
            the losses for this training iteration
        """
        year = batch.get("year")
        if year:
            noise = torch.cat((noise, year), 1)
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()
        with torch.no_grad():
            fake_images = self.generator(
                noise, block=self.block, alpha=self.alpha
            )

        fake_pred = self.discriminator(
            images=fake_images,
            year=year,
            block=self.block,
            alpha=self.alpha,
        )
        real_pred = self.discriminator(
            images=batch.get("images"),
            year=year,
            block=self.block,
            alpha=self.alpha,
        )
        loss = torch.mean(fake_pred) - torch.mean(real_pred)
        fake_batch = {"images": fake_images, "year": batch.get("year")}
        gp = self._gradient_penalty(
            batch, fake_batch, block=self.block, alpha=self.alpha
        )
        loss += gp
        loss += 0.001 * torch.mean(real_pred ** 2)
        self.manual_backward(loss)
        discriminator_optimizer.step()
        return loss

    def train_generator(
        self, batch: Dict[str, torch.Tensor], noise: torch.Tensor
    ) -> List:
        """
        Trains the generator on a single batch of data and returns the
        Generator loss.

        Args:
            batch: Batch of real data to train on
            noise: Randomly generated noise
        """
        year = batch.get("year")
        if year:
            noise = torch.cat((noise, year), 1)
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()
        generated_images = self.generator(
            noise, block=self.block, alpha=self.alpha
        )
        fake_pred = self.discriminator(
            images=generated_images,
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        loss_fake = -torch.mean(fake_pred)
        self.manual_backward(loss_fake)
        generator_optimizer.step()
        return loss_fake.detach().cpu().numpy().tolist()
