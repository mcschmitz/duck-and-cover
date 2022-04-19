from typing import Any, Dict

import torch
from torch import Tensor, nn

from networks.utils import calc_channels_at_stage
from networks.utils.layers import (
    MinibatchStdDev,
    PixelwiseNorm,
    ScaledConv2d,
    ScaledConv2dTranspose,
    ScaledDense,
)


class ProGANDiscriminatorFinalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_release_year: bool = False,
    ):
        """
        Final block for the Discriminator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            add_release_year: Flag to add the year information of the cover
        """
        super(ProGANDiscriminatorFinalBlock, self).__init__()
        final_block_in_channels = in_channels + 1
        self.add_release_year = add_release_year
        if self.add_release_year:
            final_block_in_channels += 1

        self.conv_1 = ScaledConv2d(
            final_block_in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.dense = ScaledDense(out_channels, 1, bias=True, gain=1)

    def forward(self, images: Tensor, year: Tensor = None) -> Tensor:
        """
        Forward pass of the module.

        Args:
            images: Input tensor of images
            year: Input tensor containing the release year of the images
        """
        x = MinibatchStdDev()(images)
        if self.add_release_year:
            o = torch.ones_like(x[:, 0, :, :])
            year_channel = torch.stack([oi * yi for oi, yi in zip(o, year)])
            year_channel = year_channel.reshape(-1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, year_channel], 1)
        x = self.conv_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv_2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = nn.Flatten()(x)
        return self.dense(x)


class ProGANDiscriminatorGeneralBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        General block used in the discriminator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ProGANDiscriminatorGeneralBlock, self).__init__()

        self.conv_1 = ScaledConv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the general block.

        Args:
            x: Input tensor
        """
        x = self.conv_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv_2(x)
        x = nn.LeakyReLU(0.2)(x)
        return nn.AvgPool2d(2)(x)


class ProGANDiscriminator(nn.Module):
    def __init__(
        self,
        n_blocks: int = 7,
        n_channels: int = 3,
        latent_size: int = 512,
        add_release_year: bool = False,
    ):
        """
        Builds the ProGAN discriminator.

        Args:
            n_blocks: Number of blocks.
            n_channels: Number of input channels
            latent_size: Latent size of the corresponding generator
            add_release_year: Flag to take year information into
                consideration during discrimination

        References:
            - Progressive Growing of GANs for Improved Quality, Stability, and Variation: https://arxiv.org/abs/1710.10196
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.num_channels = n_channels
        self.latent_size = latent_size

        self.layers = nn.ModuleList()
        for block in range(0, n_blocks):
            self.layers.append(
                ProGANDiscriminatorGeneralBlock(
                    calc_channels_at_stage(block + 1),
                    calc_channels_at_stage(block),
                ),
            )

        self.layers.append(
            ProGANDiscriminatorFinalBlock(
                calc_channels_at_stage(0),
                latent_size,
                add_release_year=add_release_year,
            )
        )
        self.from_rgb = nn.ModuleList(
            [
                nn.Sequential(
                    ScaledConv2d(
                        n_channels,
                        calc_channels_at_stage(stage),
                        kernel_size=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
                for stage in range(0, n_blocks)
            ]
        )

    def forward(
        self,
        images: Tensor,
        year: Tensor = None,
        block: int = 0,
        alpha: float = 1.0,
    ) -> Tensor:
        """
        Forward pass of the ProGAN discriminator.

        Args:
            images: Input tensor of images
            year: Standardized input tensor of the release year
            block: Output block
            alpha: Weight for average with the next block
        """
        if block > self.n_blocks:
            raise ValueError(
                f"This model only has {self.n_blocks} blocks. depth parameter has to be <= n_blocks"
            )
        if block == 0:
            x = self.from_rgb[0](images)
        else:
            residual = self.from_rgb[block - 1](
                nn.functional.avg_pool2d(images, kernel_size=2, stride=2)
            )
            straight = self.layers[block - 1](self.from_rgb[block](images))
            x = (alpha * straight) + ((1 - alpha) * residual)

            for layer_block in reversed(self.layers[: block - 1]):
                x = layer_block(x)
        return self.layers[-1](images=x, year=year)

    def get_save_info(self) -> Dict[str, Any]:
        """
        Collects the info about the Discriminator when writing the model.
        """
        return {
            "conf": {
                "depth": self.n_blocks,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size,
                "num_classes": self.num_classes,
            },
            "state_dict": self.state_dict(),
        }


class GenInitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Module implementing the initial block of the input.

        Args:
            in_channels: number of input channels to the block
            out_channels: number of output channels of the block
        """
        super(GenInitialBlock, self).__init__()

        self.conv_1 = ScaledConv2dTranspose(
            in_channels,
            out_channels,
            kernel_size=4,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the initial generator block.

        Args:
            x: Input Tensor
        """
        x = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        x = self.conv_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = PixelwiseNorm()(x)
        x = self.conv_2(x)
        x = nn.LeakyReLU(0.2)(x)
        return PixelwiseNorm()(x)


class GenGeneralConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        General generator block.

        Args:
            in_channels: Number of input channels to the block
            out_channels: Number of output channels required
        """
        super(GenGeneralConvBlock, self).__init__()

        self.conv_1 = ScaledConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the general generator block.

        Args:
            x: Input tensor
        """
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv_1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = PixelwiseNorm()(x)
        x = self.conv_2(x)
        x = nn.LeakyReLU(0.2)(x)
        return PixelwiseNorm()(x)


class ProGANGenerator(nn.Module):
    def __init__(
        self,
        n_blocks: int = 10,
        n_channels: int = 3,
        latent_size: int = 512,
        add_release_year: bool = False,
    ):
        """
        Generator Model of the ProGAN network.

        Args:
            n_blocks: Depth of the network
            n_channels: Number of output channels (default = 3 for RGB)
            latent_size: Latent space dimensions
            add_release_year: Boolean indicating whether to add release year
        """
        super().__init__()
        if add_release_year:
            latent_size += 1

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
                )
            )

        self.layers.append(
            ScaledConv2d(
                calc_channels_at_stage(n_blocks),
                n_channels,
                kernel_size=(1, 1),
                gain=1,
            )
        )

        self.rgb_converters = nn.ModuleList(
            [
                ScaledConv2d(
                    calc_channels_at_stage(stage),
                    n_channels,
                    kernel_size=(1, 1),
                )
                for stage in range(0, n_blocks)
            ]
        )

    def forward(
        self,
        x: Tensor,
        year: Tensor = None,
        block: int = 0,
        alpha: float = 1.0,
    ) -> Tensor:
        """
        Forward pass of the Generator.

        Args:
            x: input latent noise
            block: depth from where the network's output is required
            alpha: value of alpha for fade-in effect
            year: Standardized release year
        """
        if year is not None:
            x = torch.cat([x, year], dim=1)
        if block > self.n_blocks:
            raise ValueError(
                f"This model only has {self.n_blocks} blocks. depth parameter has to be <= n_blocks"
            )

        if block == 0:
            return self.rgb_converters[0](self.layers[0](x))
        for layer_block in self.layers[:block]:
            x = layer_block(x)
        residual = nn.functional.interpolate(
            self.rgb_converters[block - 1](x), scale_factor=2
        )
        straight = self.rgb_converters[block](self.layers[block](x))
        return (alpha * straight) + ((1 - alpha) * residual)

    def get_save_info(self) -> Dict[str, Any]:
        """
        Collects info about the Generator when saving it.
        """
        return {
            "conf": {
                "depth": self.n_blocks,
                "num_channels": self.n_channels,
                "latent_size": self.latent_size,
                "use_eql": self.use_eql,
            },
            "state_dict": self.state_dict(),
        }
