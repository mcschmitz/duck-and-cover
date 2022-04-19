import random

import numpy as np
import torch
from torch import nn

from networks.modules.stylegan import (
    GenGeneralConvBlock,
    GenInitialBlock,
    StyleGANMappingNetwork,
    Truncation,
)
from networks.utils import calc_channels_at_stage
from networks.utils.layers import ScaledConv2d


class GSynthesis(nn.Module):
    def __init__(
        self,
        n_blocks: int = 10,
        latent_size: int = 512,
        n_channels: int = 3,
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.latent_size = latent_size
        self.n_channels = n_channels

        self.layers = nn.ModuleList()
        self.layers.append(
            GenInitialBlock(
                latent_size,
                calc_channels_at_stage(0),
            )
        )

        for block in range(0, n_blocks):
            self.layers.append(
                GenGeneralConvBlock(
                    calc_channels_at_stage(block),
                    calc_channels_at_stage(block + 1),
                    dlatent_size=latent_size,
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
        x = self.init_block(w[:, 0:2])

        if block == 0:
            return self.rgb_converters[0](self.layers[0](x))
        for layer_block in self.layers[:block]:
            x = layer_block(x)
        residual = nn.functional.interpolate(
            self.rgb_converters[block - 1](x), scale_factor=2
        )
        straight = self.rgb_converters[block](self.layers[block](x))
        return (alpha * straight) + ((1 - alpha) * residual)


class Generator(nn.Module):
    def __init__(
        self,
        n_blocks: int = 10,
        latent_size: int = 512,
        n_channels: int = 3,
        style_mixing_prob=0.9,
    ):
        super(Generator, self).__init__()

        self.style_mixing_prob = style_mixing_prob

        self.truncation = Truncation(
            avg_latent=torch.zeros(latent_size),
            max_layer=8,
            threshold=0.7,
            beta=0.995,
        )

        self.g_mapping = StyleGANMappingNetwork(latent_size, 8)
        self.g_synthesis = GSynthesis(
            n_blocks=n_blocks, latent_size=latent_size, n_channels=n_channels
        )

    def forward(self, x, block: int, alpha: float):
        w = self.g_mapping(x)

        if self.training:
            latents2 = torch.randn(x.shape).to(x.device)
            dlatents2 = self.g_mapping(latents2)
            layer_idx = torch.from_numpy(
                np.arange(self.num_layers)[np.newaxis, :, np.newaxis]
            ).to(x.device)
            cur_layers = 2 * (block + 1)
            mixing_cutoff = (
                random.randint(1, cur_layers)
                if random.random() < self.style_mixing_prob
                else cur_layers
            )
            w = torch.where(layer_idx < mixing_cutoff, w, dlatents2)

            w = self.truncation(w)

        return self.g_synthesis(w, block, alpha)
