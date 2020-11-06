"""
Abstract Base class for generating Cover GANS.
"""
from abc import ABC, abstractmethod


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
        self.history = {}

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
        self.history["G_loss"] = []
