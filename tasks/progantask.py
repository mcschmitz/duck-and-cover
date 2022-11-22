from copy import deepcopy
from typing import Dict, List

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from config import GANTrainConfig
from tasks.wgantask import WGANTask
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
        self.alpha_tick = None
        self.ema_generator = deepcopy(self.generator)
        self.update_ema_generator(0.0)
        self.current_resolution = 2 ** (self.block + 2)

        self.upsample = self.generator.upsample
        self.downsample = self.discriminator.downsample

    def on_fit_start(self):
        """
        Gets the current image size and initializes the alpha weights for the
        fade in phase.
        """
        super().on_fit_start()
        self.current_resolution = 2 ** (self.block + 2)
        if self.phase == "fade_in":
            if self.alpha_tick is None:
                self.alpha_tick = 1 / (self.trainer.max_steps / 2)
            logger.info(
                f"Phase: Fade in for resolution {self.current_resolution}"
            )
        else:
            logger.info(
                f"Phase: Burn in for resolution {self.current_resolution}"
            )

    def configure_callbacks(self) -> List[pl.Callback]:
        """
        Creates the callbacks for the training.

        Will add a ModelCheckpoint Callback to save the model at every n
        steps (n has to be defined in the config that is passed during
        model initialization) and a GenerateImages callback to generate
        images at every n steps.
        """
        self.phase_steps = 0
        every_n_train_steps = (
            self.trainer.max_steps - (self.phase_steps * 2)
        ) // self.config.n_evals
        return self._configure_callbacks(every_n_train_steps)

    def training_step(self, batch: Dict, batch_idx: int):
        """
        Trains the ProGAN on a single batch of data. Obtains the phase (`burn-
        in` or `fade-in`) and trains the GAN accordingly.

        Args:
            batch: Batch to train on
            batch_idx: Incremental batch index
        """
        self.logger.log_metrics(
            {"trainer/alpha": self.alpha}, step=self.images_shown
        )
        if self.phase == "fade_in":
            self.alpha += self.alpha_tick
            self.alpha = np.clip(self.alpha, 0, 1)
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

    def downscale_images(self, images: Tensor):
        while images.shape[-1] != self.current_resolution:
            images = self.downsample(images)
        images_lowres = self.upsample(self.downsample(images))
        return images * self.alpha + images_lowres * (1 - self.alpha)

    def train_on_batch(self, batch: Dict[str, Tensor], batch_idx: int):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Batch of data to train on
            batch_idx: Incremental batch index
        """
        batch["images"] = self.downscale_images(batch["images"])
        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        noise = noise.to(self.device)
        with torch.no_grad():
            generated_images = self.generator(
                x=noise,
                year=batch.get("year"),
                block=self.block,
                alpha=self.alpha,
            )
        self.train_discriminator(batch, generated_images)
        if batch_idx % self.config.n_critic == 0:
            self.train_generator(batch=batch, noise=noise)

    def train_discriminator(
        self, batch: Dict[str, torch.Tensor], generated_images: Tensor
    ):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Real input images used for training
            generated_images: Images generated by the generator
        """
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()

        real_pred = self.discriminator(
            images=batch["images"],
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        fake_pred = self.discriminator(
            images=generated_images,
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )

        loss = torch.mean(fake_pred) - torch.mean(real_pred)
        gp = self._gradient_penalty(
            batch, generated_images, block=self.block, alpha=self.alpha
        )
        loss += self.config.gradient_penalty_weight * gp
        self.manual_backward(loss)
        discriminator_optimizer.step()

        batch_size = generated_images.shape[0]
        real = torch.zeros(batch_size).to(self.device)
        fake = torch.ones(batch_size).to(self.device)
        fake_pred = self.sigmoid(fake_pred)
        real_pred = self.sigmoid(real_pred)
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        discriminator_acc = (correct_real + correct_fake) / (batch_size * 2)
        self.logger.log_metrics(
            {
                "train/discriminator_loss": loss,
                "train/discriminator_acc": discriminator_acc,
            },
            step=self.images_shown,
        )

    def train_generator(
        self, batch: Dict[str, torch.Tensor], noise: torch.Tensor
    ):
        """
        Trains the generator on a single batch of data and returns the
        Generator loss.

        Args:
            batch: Input batch containing images and year
            noise: Randomly generated noise
        """
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()
        noise = noise.to(self.device)
        generated_images = self.generator(
            x=noise, year=batch.get("year"), block=self.block, alpha=self.alpha
        )
        g_loss = torch.mean(
            -self.discriminator(
                images=generated_images,
                year=batch.get("year"),
                block=self.block,
                alpha=self.alpha,
            )
        )
        self.manual_backward(g_loss)
        generator_optimizer.step()
        self.logger.log_metrics(
            {"train/generator_loss": g_loss}, step=self.images_shown
        )
        self.update_ema_generator(self.config.ema_beta)

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
        checkpoint["alpha_tick"] = self.alpha_tick
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
        self.alpha_tick = None  # checkpoint["alpha_tick"]
        self.phase_steps = checkpoint["phase_steps"]

    def on_fit_end(self):
        """
        After the fit of one phase switches to the next block or the burn-in
        phase and resets the phrase steps attribute.
        """
        super().on_fit_end()
        self.trainer.save_checkpoint(
            f"{self.config.learning_progress_path}/{self.current_resolution}_{self.phase}"
        )
        if self.phase == "burn_in":
            self.phase = "fade_in"
            self.phase_steps = 0
            self.block += 1
            self.alpha = 0
            self.trainer.optimizers = self.configure_optimizers()
        elif self.phase == "fade_in":
            self.phase = "burn_in"
            self.phase_steps = 0
            self.alpha_tick = None
            self.alpha = 1
        self.phase_steps = 0

    def update_ema_generator(self, beta):
        """
        After every weight update of the generator this method updates the
        Exponential Moving Average of the generator weights and stores them as
        a separate Generator model.

        This separate model shall be used for image generation.
        """

        def toggle_grad(model, requires_grad):
            for p in model.parameters():
                p.requires_grad_(requires_grad)

        # turn off gradient calculation
        toggle_grad(self.ema_generator, False)
        toggle_grad(self.generator, False)

        param_dict_src = dict(self.generator.named_parameters())

        for p_name, p_tgt in self.ema_generator.named_parameters():
            p_src = param_dict_src[p_name]
            assert p_src is not p_tgt
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

        # turn back on the gradient calculation
        toggle_grad(self.ema_generator, True)
        toggle_grad(self.generator, True)
