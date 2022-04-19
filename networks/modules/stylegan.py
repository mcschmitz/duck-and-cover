import numpy as np
import torch
from torch import Tensor, nn

from networks.utils.layers import PixelwiseNorm, ScaledConv2d, ScaledDense


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


class Epilogue(nn.Module):
    def __init__(self, n_channels, latent_size):
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
            x = layer(x, w)
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


class GenInitialBlock(nn.Module):
    def __init__(self, n_channels, latent_size):
        super().__init__()
        self.n_channels = n_channels

        self.const = nn.Parameter(torch.ones(1, n_channels, 4, 4))
        self.bias = nn.Parameter(torch.ones(n_channels))

        self.epilogue1 = Epilogue(n_channels, latent_size)
        self.conv = ScaledConv2d(n_channels, n_channels, 3)
        self.epilogue2 = Epilogue(n_channels, latent_size)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(
                batch_size, self.n_channels, 4, 4
            )

        x = self.epilogue1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epilogue2(x, dlatents_in_range[:, 1])

        return x


class GenGeneralConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dlatent_size):
        super().__init__()

        self.conv0_up = ScaledConv2d(in_channels, out_channels, kernel_size=3)
        self.epi1 = Epilogue(out_channels, dlatent_size)
        self.conv1 = ScaledConv2d(in_channels, out_channels, kernel_size=3)
        self.epi2 = Epilogue(out_channels, dlatent_size)

    def forward(self, x, dlatents_in_range):
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer("avg_latent", avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(
            self.beta * self.avg_latent + (1.0 - self.beta) * last_avg
        )

    def forward(self, x):
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (
            (torch.arange(x.size(1)) < self.max_layer)
            .view(1, -1, 1)
            .to(x.device)
        )
        return torch.where(do_trunc, interp, x)
