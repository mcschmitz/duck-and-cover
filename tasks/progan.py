from typing import Dict

import numpy as np
import torch
from torch import Tensor, nn

from config import GANTrainConfig
from tasks.wgan import WGANTask
from utils import logger


class ProGANTask(WGANTask):
    def __init__(
        self,
        config: GANTrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        ProGAN task constructor. Takes a ProGAN generator and discriminator to
        train them on the given task.

        Args:
            config: Config object containing the model hyperparameters
            generator: ProGAN generator to train
            discriminator: ProGAN discriminator to train
        """
        super().__init__(
            config=config,
            generator=generator,
            discriminator=discriminator,
        )
        self.block = 0
        self.burn_in_images_shown = 0
        self.fade_in_images_shown = 0
        self.phase_steps = 0
        self.phase = "burn_in"
        self.alpha = 1
        self.automatic_optimization = False
        self._alphas = None

    def on_fit_start(self):
        """
        Gets the current image size and initializes the alpha weights for the
        fade in phase.
        """
        super().on_fit_start()
        img_resolution = 2 ** (self.block + 2)
        if self.phase == "fade_in":
            if not self._alphas:
                self._alphas = np.linspace(
                    0,
                    1,
                    int(
                        self.trainer.max_steps / (1 + 1 / self.config.n_critic)
                    ),
                ).tolist()
            logger.info(f"Phase: Fade in for resolution {img_resolution}")
        else:
            logger.info(f"Phase: Burn in for resolution {img_resolution}")

    def training_step(self, batch: Dict, batch_idx: int):
        """
        Trains the ProGAN on a single batch of data. Obtains the phase (`burn-
        in` or `phade-in`) and trains the GAN accordingly.

        Args:
            batch: Batch to train on
            batch_idx: Incremental batch index
        """
        if self.phase == "fade_in":
            self.alpha = self._alphas.pop(0)
        self.train_on_batch(batch=batch, batch_idx=batch_idx)
        if self.phase == "fade_in":
            self.fade_in_images_shown += len(batch.get("images"))
        elif self.phase == "burn_in":
            self.burn_in_images_shown += len(batch.get("images"))
        self.log(
            "train/images_shown",
            self.fade_in_images_shown + self.burn_in_images_shown,
        )
        self.phase_steps += 1

    def train_on_batch(self, batch: Dict[str, Tensor], batch_idx: int):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Batch of data to train on
            batch_idx: Incremental batch index
        """
        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        noise = noise.to(self.device)
        generated_images = self.generator(
            noise, block=self.block, alpha=self.alpha
        )
        self.train_discriminator(batch, generated_images)
        if batch_idx % self.config.n_critic == 0:
            self.train_generator(noise)

    def train_discriminator(
        self, batch: Dict[str, torch.Tensor], generated_images: Tensor
    ):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Real input images used for training

        Returns:
            the losses for this training iteration
        """
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()

        real_pred = self.discriminator(
            batch["images"], block=self.block, alpha=self.alpha
        )
        fake_pred = self.discriminator(
            generated_images, block=self.block, alpha=self.alpha
        )

        loss = torch.mean(fake_pred) - torch.mean(real_pred)
        gp = self._gradient_penalty(
            batch, generated_images, block=self.block, alpha=self.alpha
        )
        loss += self.config.gradient_penalty_weight * gp
        self.manual_backward(loss)
        discriminator_optimizer.step()

        real = torch.zeros((self.config.batch_size, 1)).to(self.device)
        fake = torch.ones((self.config.batch_size, 1)).to(self.device)
        fake_pred = self.sigmoid(fake_pred)
        real_pred = self.sigmoid(real_pred)
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        discriminator_acc = (correct_real + correct_fake) / (
            self.config.batch_size * 2
        )
        self.log("train/discriminator_loss", loss)
        self.log("train/discriminator_accuracy", discriminator_acc)

    def train_generator(self, noise: torch.Tensor):
        """
        Trains the generator on a single batch of data and returns the
        Generator loss.

        Args:
            noise: Randomly generated noise
        """
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()
        noise = noise.to(self.device)
        generated_images = self.generator(
            noise, block=self.block, alpha=self.alpha
        )
        g_loss = torch.mean(
            -self.discriminator(
                generated_images, block=self.block, alpha=self.alpha
            )
        )
        self.manual_backward(g_loss)
        generator_optimizer.step()
        self.log("train/generator_loss", g_loss)

    def on_save_checkpoint(self, checkpoint):
        """
        Writes additional attributes to the checkpoint.

        Args:
            checkpoint: PT Lightning Checkpoint
        """
        super(ProGANTask, self).on_save_checkpoint(checkpoint)
        checkpoint["block"] = self.block
        checkpoint["phase"] = self.phase
        checkpoint["burn_in_images_shown"] = self.burn_in_images_shown
        checkpoint["fade_in_images_shown"] = self.fade_in_images_shown
        checkpoint["alpha"] = self.alpha
        checkpoint["_alphas"] = self._alphas
        checkpoint["phase_steps"] = self.phase_steps

    def on_load_checkpoint(self, checkpoint):
        """
        Loads all the additional attributes from the checkpoint.

        Args:
            checkpoint: PT Lightning Checkpoint
        """
        super(ProGANTask, self).on_load_checkpoint(checkpoint)
        self.block = checkpoint["block"]
        self.phase = checkpoint["phase"]
        self.burn_in_images_shown = checkpoint["burn_in_images_shown"]
        self.fade_in_images_shown = checkpoint["fade_in_images_shown"]
        self.alpha = checkpoint["alpha"]
        self._alphas = checkpoint["_alphas"]
        self.phase_steps = checkpoint["phase_steps"]

    def on_fit_end(self):
        """
        After the fit of one phase switches to the next block or the burn-in
        phase and resets the phrase steps attribute.
        """
        super().on_fit_end()
        if self.phase == "burn_in":
            self.phase = "fade_in"
            self.block += 1
        elif self.phase == "fade_in":
            self.phase = "burn_in"
        self.phase_steps = 0
