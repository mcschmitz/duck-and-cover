import random

import numpy as np
import torch
from torch import Tensor, nn

from networks.modules.progan import ProGANDiscriminator
from networks.modules.stylegan import (
    StyleGANGenGeneralConvBlock,
    StyleGANGenInitialBlock,
    StyleGANMappingNetwork,
)
from networks.progan import ProGAN
from networks.utils import calc_channels_at_stage
from networks.utils.layers import ScaledConv2d, Truncation


class StyleGANSynthesis(nn.Module):
    def __init__(
        self,
        n_blocks: int = 10,
        latent_size: int = 512,
        n_channels: int = 3,
    ):
        super().__init__()

        self.n_blocks = n_blocks

        self.layers = nn.ModuleList()
        self.layers.append(
            StyleGANGenInitialBlock(calc_channels_at_stage(0), latent_size)
        )

        for block in range(0, n_blocks):
            self.layers.append(
                StyleGANGenGeneralConvBlock(
                    calc_channels_at_stage(block),
                    calc_channels_at_stage(block + 1),
                    latent_size=latent_size,
                )
            )

        self.rgb_converters = nn.ModuleList(
            [
                ScaledConv2d(
                    calc_channels_at_stage(stage),
                    n_channels,
                    kernel_size=(1, 1),
                    gain=1.0,
                )
                for stage in range(0, n_blocks)
            ]
        )

    def forward(self, w, block: int = 0, alpha: float = 0.0):
        if block > self.n_blocks:
            raise ValueError(
                f"This model only has {self.n_blocks} blocks. depth parameter has to be <= n_blocks"
            )
        x = self.layers[0](w[:, 0:2])
        if block == 0:
            return self.rgb_converters[0](x)
        for i, layer_block in enumerate(self.layers[:block][1:]):
            x = layer_block(x, w[:, (i + 1) : (i + 3)])
        residual = nn.functional.interpolate(
            self.rgb_converters[block - 1](x), scale_factor=2
        )
        straight = self.rgb_converters[block](self.layers[block](x, w))
        return (alpha * straight) + ((1 - alpha) * residual)


class StyleGANGenerator(nn.Module):
    def __init__(
        self,
        n_blocks: int = 10,
        latent_size: int = 512,
        n_channels: int = 3,
        style_mixing_prob=0.9,
    ):
        super(StyleGANGenerator, self).__init__()

        self.n_blocks = n_blocks
        self.latent_size = latent_size
        self.n_channels = n_channels

        self.style_mixing_prob = style_mixing_prob
        self.truncation = Truncation(
            avg_latent=torch.zeros(latent_size),
            max_layer=8,
            threshold=0.7,
            beta=0.995,
        )

        self.g_mapping = StyleGANMappingNetwork(
            latent_size, 8, n_blocks=self.n_blocks + 1
        )
        self.g_synthesis = StyleGANSynthesis(
            n_blocks=n_blocks, latent_size=latent_size, n_channels=n_channels
        )

    def forward(
        self,
        x: Tensor,
        year: Tensor = None,
        block: int = 0,
        alpha: float = 1.0,
    ):
        w_orig = self.g_mapping(x)

        if self.training:
            x_rand = torch.randn(x.shape).to(x.device)
            w_rand = self.g_mapping(x_rand)
            layer_idx = torch.from_numpy(
                np.arange(self.g_mapping.n_blocks)[np.newaxis, :, np.newaxis]
            ).to(x.device)
            cur_layers = 2 * (block + 1)
            mixing_cutoff = (
                random.randint(1, cur_layers)
                if random.random() < self.style_mixing_prob
                else cur_layers
            )
            w_orig = torch.where(layer_idx < mixing_cutoff, w_orig, w_rand)

            w_orig = self.truncation(w_orig)

        return self.g_synthesis(w_orig, block, alpha)


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
        )

    def build_generator(self) -> StyleGANGenerator:
        """
        Builds the ProGAN Generator.
        """
        return StyleGANGenerator(
            n_blocks=self.config.n_blocks,
            n_channels=self.config.channels,
            latent_size=self.config.latent_size,
        )
