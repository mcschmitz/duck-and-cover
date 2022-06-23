from torch import nn

from networks.modules.progan import ProGANDiscriminator
from networks.modules.stylegan import StyleGANGenerator
from networks.progan import ProGAN


class StyleGAN(ProGAN):
    def build_discriminator(self) -> ProGANDiscriminator:
        """
        Builds the ProGAN Discriminator.
        """
        return ProGANDiscriminator(
            n_blocks=self.config.n_blocks,
            n_channels=self.config.channels,
            latent_size=self.config.latent_size,
            add_release_year=self.config.add_release_year,
            downsample=nn.UpsamplingBilinear2d(scale_factor=0.5),
        )

    def build_generator(self) -> StyleGANGenerator:
        """
        Builds the ProGAN Generator.
        """
        return StyleGANGenerator(
            n_blocks=self.config.n_blocks,
            n_channels=self.config.channels,
            latent_size=self.config.latent_size,
            n_mapping=self.config.n_mapping,
            style_mixing_prob=self.config.style_mixing_prob,
        )
