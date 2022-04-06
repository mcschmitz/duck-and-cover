import copy
import os

import joblib
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

from config import GANTrainConfig
from utils import logger


class CoverGANTask(pl.LightningModule):
    def __init__(
        self,
        config: GANTrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Task to train a GAN.

        Args:
            config: Training configuration object.
            generator: PyTorch module that generates images.
            discriminator: PyTorch module that discriminates between real and
                generated images.
        """
        super(CoverGANTask, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.config = config
        self.wandb_run_id = None
        self.images_shown = 0
        self.automatic_optimization = False

    def configure_optimizers(self):
        """
        Assign both the discriminator and the generator optimizer.
        """
        discriminator_optimizer = Adam(
            params=self.discriminator.parameters(), lr=self.config.disc_lr, betas=self.config.disc_betas
        )
        generator_optimizer = Adam(
            self.generator.parameters(), lr=self.config.gen_lr, betas=self.config.gen_betas
        )
        return generator_optimizer, discriminator_optimizer

    def save(self, path: str):
        """
        Saves the weights of the Cover GAN.

        Writes the weights of the GAN object to the given path. The combined
        model weights, the discriminator weights and the generator weights
        will be written separately to the given directory

        Args:
            path: The directory to which the weights should be written.
        """
        torch.save(
            {
                "generator_state_dict": self.generator.state_dict(),
                "generator_optimizer": self.generator_optimizer.state_dict(),
            },
            os.path.join(path, "generator.pkl"),
        )
        torch.save(
            {
                "discriminator_state_dict": self.discriminator.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            },
            os.path.join(path, "discriminator.pkl"),
        )

        gan = copy.copy(self)
        gan.discriminator = None
        gan.generator = None
        gan.generator_optimizer = None
        gan.discriminator_optimizer = None
        joblib.dump(gan, os.path.join(path, "GAN.pkl"))

    def load(self, path):
        """
        Load the weights and the attributes of the GAN.

        Loads the weights and the pickle object written in the save method.

        Args:
            path: The directory from which the weights and the GAN should be
                read.
        """
        logger.info(f"Loading weights & optimizer from {path}")
        generator_cktp = torch.load(os.path.join(path, "generator.pkl"))
        discriminator_cktp = torch.load(
            os.path.join(path, "discriminator.pkl")
        )
        gan = joblib.load(os.path.join(path, "GAN.pkl"))
        self.images_shown = gan.images_shown
        self.metrics = gan.metrics
        self.generator.load_state_dict(generator_cktp["generator_state_dict"])
        self.generator_optimizer.load_state_dict(
            generator_cktp["generator_optimizer"]
        )
        self.discriminator.load_state_dict(
            discriminator_cktp["discriminator_state_dict"]
        )
        self.discriminator_optimizer.load_state_dict(
            discriminator_cktp["discriminator_optimizer"]
        )
        return gan

    def on_fit_start(self):
        """
        This method gets executed before a Trainer trains this model.

        It tells the W&B logger to watch the model in order to check the
        gradients report the gradients if W&B is online.
        """
        if hasattr(self, "logger"):
            if isinstance(self.logger, pl.loggers.WandbLogger):
                try:
                    self.logger.watch(self)
                except ValueError:
                    logger.info("The model is already on the watchlist")
                self.wandb_run_id = self.logger.experiment.id
