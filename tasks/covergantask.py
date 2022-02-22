import copy
import os
from abc import abstractmethod

import joblib
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Adam

from utils import logger


class CoverGANTask(pl.LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        name: str,
        **kwargs,
    ):
        """
        Abstract GAN Class.

        Args:
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the
                image
            use_gpu: Flag to use the GPU for training and prediction
        """
        super(CoverGANTask, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

        self.wandb_run_id = None
        self.wandb_run_name = name

    def configure_optimizers(self):
        """
        Assign both the discriminator and the generator optimizer.
        """
        discriminator_optimizer = Adam(
            params=self.discriminator.parameters(), lr=0.001, betas=(0.0, 0.99)
        )
        generator_optimizer = Adam(
            self.generator.parameters(), lr=0.001, betas=(0.0, 0.99)
        )
        return generator_optimizer, discriminator_optimizer

    @abstractmethod
    def build_generator(self, **kwargs):
        """
        Builds the class specific generator.
        """

    @abstractmethod
    def build_discriminator(self, **kwargs):
        """
        Builds the class specific discriminator.
        """

    @abstractmethod
    def train_on_batch(self, *args):
        """
        Abstract method to train the combined model on a batch of data.
        """

    @abstractmethod
    def train_discriminator(self, *args):
        """
        Abstract method to train the discriminator on a batch of data.
        """

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
                self.logger.watch(self)
                self.wandb_run_id = self.logger.experiment.id
                self.wandb_run_name = self.logger.experiment.name
