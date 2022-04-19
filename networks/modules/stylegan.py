import numpy as np
import torch
from torch import Tensor, nn

from networks.utils.layers import (
    PixelwiseNorm,
    ScaledConv2d,
    ScaledConv2dTranspose,
    ScaledDense,
)


class StyleGANMappingNetwork(nn.Module):
    def __init__(
        self, latent_size: int, mapping_layers: int, n_blocks: int = 10
    ):
        """
        StyleGan Mapping Network as described in the paper `A Style-Based
        Generator Architecture for Generative Adversarial Networks`

        Args:
            latent_size: Size of the latent space
            mapping_layers: Number of layers in the mapping network
            n_blocks: Total Number of blocks the Generator is composed of

        References:
            https://arxiv.org/pdf/1812.04948.pdf
        """
        super().__init__()
        self.n_blocks = n_blocks
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
        return x.unsqueeze(1).expand(-1, self.n_blocks, -1)


class Epilogue(nn.Module):
    def __init__(self, n_channels: int, latent_size: int):
        """
        Epilogue block that gets executed after every Conv2d layer.

        Args:
            n_channels: Number of channels
            latent_size: Latent space size
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                NoiseLayer(n_channels),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(n_channels),
            ]
        )
        self.ada_in = AdaIn(latent_size, n_channels)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Passes the input tensor through the layers and mixes in the latent
        vector w form the mapping network.

        Args:
            x: Input tensor from the last Conv2d layer
            w: Latent vector from the mapping network
        """
        for layer in self.layers:
            x = layer(x)
        return self.ada_in(x, w)


class NoiseLayer(nn.Module):
    def __init__(self, n_channels: int):
        """
        Injects noise into the input tensor.

        Args:
            n_channels: Number of channels
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        Randomly generates noise and mixes it with the input tensor.

        Args:
            x: Input tensor
        """
        noise = torch.randn(
            x.size(0),
            1,
            x.size(2),
            x.size(3),
            device=x.device,
            dtype=x.dtype,
        )
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class AdaIn(nn.Module):
    def __init__(self, latent_size: int, n_channels: int):
        """
        StyleGAN adaptive instance normalization.

        Args:
            latent_size: Size of the latent space
            n_channels: Number of channels
        """
        super(AdaIn, self).__init__()
        self.dense = ScaledDense(latent_size, n_channels * 2, gain=1.0)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Each feature map xi is normalized separately, and then scaled and
        biased using the corresponding scalar components from style w.

        Args:
            x: Input tensor
            w: Latent vector from the mapping network
        """
        style = self.dense(w)
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class StyleGANGenInitialBlock(nn.Module):
    def __init__(self, n_channels: int, latent_size: int):
        """
        Initial block of the StyleGAN generator.

        Args:
            n_channels: Number of channels of the image
            latent_size: Dimensionality of the latent space
        """
        super().__init__()
        self.n_channels = n_channels

        self.const = nn.Parameter(torch.ones(1, n_channels, 4, 4))
        self.bias = nn.Parameter(torch.ones(n_channels))

        self.epilogue_1 = Epilogue(n_channels, latent_size)
        self.conv = ScaledConv2dTranspose(
            n_channels, n_channels, kernel_size=3, padding=1
        )
        self.epilogue_2 = Epilogue(n_channels, latent_size)

    def forward(self, w: Tensor) -> Tensor:
        """
        Passes the latent vector w through the initial block of the generator.
        Specifically, the latent vector is mixed with the noise, normalized and
        then mixed with the constant input tensor (in the epilogue). After that
        a ScaledConv2d is applied to the input tensor and the epilogue is
        applied to its output.

        Args:
            w: Latent vector from the mapping network
        """
        batch_size = w.size(0)
        x = self.const.expand(batch_size, -1, -1, -1)
        x = x + self.bias.view(1, -1, 1, 1)
        x = self.epilogue_1(x, w[:, 0])
        x = self.conv(x)
        return self.epilogue_2(x, w[:, 1])


class StyleGANGenGeneralConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, latent_size: int):
        """
        General convolution block of the StyleGAN generator.

        Args:
            in_channels: Number of input channels to the block
            out_channels: Number of output channels required
            latent_size: Dimensionality of the latent space
        """
        super().__init__()

        self.conv_1 = ScaledConv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.epilogue_1 = Epilogue(out_channels, latent_size)
        self.conv_2 = ScaledConv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.epilogue_2 = Epilogue(out_channels, latent_size)

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """
        Forward pass of the general convolution block. Specifically, scales the
        input tensor up by a factor of 2, applies the first ScaledConv2d and
        then applies the epilogue operations (Mixing with noise, LeakyRelu,
        Instance Normalization and Adaptive Input Normalization with the
        latent_space). After that, applies the second ScaledConv2d and again
        runs the epilogue operations.

        Args:
            x: Output of the previous block
            w: Latent vector from the mapping network
        """
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv_1(x)
        x = self.epilogue_1(x, w[:, 0])
        x = self.conv_2(x)
        return self.epilogue_2(x, w[:, 1])
