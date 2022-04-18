import numpy as np
from torch import Tensor, nn

from networks.utils.layers import PixelwiseNorm, ScaledDense


class StyleGANMappingNetwork(nn.Module):
    def __init__(self, latent_size: int, mapping_layers: int):
        """
        StyleGan Mapping Network as described in the paper `A Style-Based
        Generator Architecture for Generative Adversarial Networks`

        Args:
            latent_size: Size of the latent space
            mapping_layers: Number of layers in the mapping network

        References:
            https://arxiv.org/pdf/1812.04948.pdf
        """
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.mapping_network = nn.ModuleList([PixelwiseNorm()])
        for _ in range(mapping_layers):
            self.mapping_network.append(
                ScaledDense(latent_size, latent_size, gain=np.sqrt(2))
            )
            self.mapping_network.append(self.lrelu)

    def forward(self, x: Tensor):
        """
        Passes the input tensor through the mapping network.

        Args:
            x: Latent space vector
        """
        for layer in self.mapping_network:
            x = layer(x)
        return x
