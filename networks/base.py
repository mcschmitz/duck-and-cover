import copy
import os
from abc import ABC, abstractmethod
from collections import defaultdict

import joblib

from utils import logger


class GAN(ABC):
    def __init__(self):
        """
        Abstract GAN Class.
        """
        self.discriminator = None
        self.generator = None
        self.discriminator_model = None
        self.combined_model = None
        self.images_shown = 0
        self.generator_loss = []
        self.metrics = defaultdict(dict)

    @abstractmethod
    def build_models(self, *args):
        """
        Abstract method to build the models for the GAN.
        """

    @abstractmethod
    def _build_discriminator(self):
        """
        Abstract method to define the discriminator architecture.
        """

    @abstractmethod
    def _build_combined_model(self, *args):
        """
        Abstract method to build the combined model.
        """

    @abstractmethod
    def _build_generator(self):
        """
        Abstract method to define the generator architecture.
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
        gan = copy.copy(self)
        gan.combined_model.save_weights(os.path.join(path, "C.h5"))
        gan.discriminator.save_weights(os.path.join(path, "D.h5"))
        gan.generator.save_weights(os.path.join(path, "G.h5"))
        gan.discriminator = None
        gan.generator = None
        gan.discriminator_model = None
        gan.combined_model = None
        joblib.dump(gan, os.path.join(path, "GAN.pkl"))

    def load_weights(self, path):
        """
        Load the weights and the attributes of the GAN.

        Loads the weights and the pickle object written in the save method.

        Args:
            path: The directory from which the weights and the GAN should be
                read.
        """
        logger.info(f"Loading weights from {path}")
        self.combined_model.load_weights(os.path.join(path, "C.h5"))
        self.discriminator.load_weights(os.path.join(path, "D.h5"))
        self.generator.load_weights(os.path.join(path, "G.h5"))
        gan = joblib.load(os.path.join(path, "GAN.pkl"))
        self.images_shown = gan.images_shown
        self.metrics = gan.metrics
