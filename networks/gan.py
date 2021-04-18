import copy
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import joblib
import torch
from torch.optim import Adam

from utils import logger


class GAN(ABC):
    def __init__(self, use_gpu: bool = False):
        """
        Abstract GAN Class.

        Args:
            use_gpu: Flag to use GPU for training
        """
        self.discriminator = None
        self.generator = None
        self.discriminator_optimizer = None
        self.generator_optimizer = None
        self.images_shown = 0
        self.generator_loss = []
        self.metrics = defaultdict(dict)
        self.use_gpu = torch.cuda.is_available() and use_gpu

    def set_optimizers(self, discriminator_optimizer, generator_optimizer):
        """
        Assign both the discriminator and the generator optimizer.

        Args:
            discriminator_optimizer: PyTorch discriminator optimizer
            generator_optimizer: PyTorch generator optimizer
        """
        self.discriminator_optimizer = Adam(
            params=self.discriminator.parameters(), **discriminator_optimizer
        )
        self.generator_optimizer = Adam(
            self.generator.parameters(), **generator_optimizer
        )

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

    def fuse_disc_and_gen(self, optimizer):
        """
        Combines discriminator and generator to one model.

        Args:
            optimizer: Optimizer  to use to train the combined model
        """
        for discriminator_layer in self.discriminator.layers:
            discriminator_layer.trainable = False
        self.discriminator.trainable = False
        for generator_layer in self.generator.layers:
            generator_layer.trainable = True
        self.generator.trainable = True
        self._build_combined_model(optimizer)

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
