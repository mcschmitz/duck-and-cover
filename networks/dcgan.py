from torch import nn

from config import GANConfig
from networks.modules.dcgan import DCDiscrimininator, DCGenerator


class DCGAN(nn.Module):
    def __init__(self, config: GANConfig):
        """
        Simple Deep Convolutional GAN.

        Simple DCGAN that builds a discriminator a generator and the adversarial
        model to train the GAN based on the binary crossentropy loss for the
        generator and the discriminator.

        Args:
            config: Config object containing the model hyperparameters
        """
        super().__init__()
        self.config = config
        self.img_shape = (
            self.config.channels,
            self.config.image_size,
            self.config.image_size,
        )

    def build_generator(self) -> DCGenerator:
        """
        Builds the class specific generator.
        """
        return DCGenerator(self.config.latent_size, self.img_shape)

    def build_discriminator(self) -> DCDiscrimininator:
        """
        Builds the class specific discriminator.
        """
        return DCDiscrimininator(self.img_shape)
