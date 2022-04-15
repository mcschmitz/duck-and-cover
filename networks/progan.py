from networks.modules.progan import ProGANDiscriminator, ProGANGenerator
from networks.wgan import WGAN


class ProGAN(WGAN):
    """
    Progressive growing GAN.

    Progressive Growing GAN that iteratively adds convolutional blocks
    to generator and discriminator. This results in improved quality,
    stability, variation and a faster learning time. Initialization
    itself is similar to a normal Wasserstein GAN. Training, however, is
    slightly different: Initialization of the training phase should be
    done on a small scale image (e.g. 4x4). After an initial burn-in
    phase (train the GAN on the current resolution), a fade-in phase
    follows; meaning that on fades in the 8x8 upscale layer using the
    add add_fade_in_layers method. This phase is followed by another
    burn-in phase on the current image resolution. The procedure is
    repeated until the output image has the desired resolution.
    """

    def build_discriminator(self) -> ProGANDiscriminator:
        """
        Builds the ProGAN Discriminator.
        """
        return ProGANDiscriminator(
            n_blocks=self.config.n_blocks,
            n_channels=self.config.channels,
            latent_size=self.config.latent_size,
            add_release_year=self.config.add_release_year,
        )

    def build_generator(self) -> ProGANGenerator:
        """
        Builds the ProGAN Generator.
        """
        return ProGANGenerator(
            n_blocks=self.config.n_blocks,
            n_channels=self.config.channels,
            latent_size=self.config.latent_size,
            add_release_year=self.config.add_release_year,
        )
