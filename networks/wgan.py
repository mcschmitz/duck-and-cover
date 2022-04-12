from networks.dcgan import DCGAN
from networks.modules.wgan import WGANDiscriminator, WGANGenerator


class WGAN(DCGAN):
    def build_generator(self) -> WGANGenerator:
        """
        Builds the WGAN Generator.
        """
        return WGANGenerator(self.config.latent_size, self.img_shape)

    def build_discriminator(self) -> WGANDiscriminator:
        """
        Builds the WGAN Discriminator.
        """
        return WGANDiscriminator(self.img_shape)
