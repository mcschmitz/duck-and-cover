import os
from typing import Any, Dict

import numpy as np
import torch
from defaultlist import defaultlist
from torch import nn

from networks.utils import calc_channels_at_stage, plot_metric
from networks.utils.layers import (
    MinibatchStdDev,
    PixelwiseNorm,
    ScaledConv2d,
    ScaledConv2dTranspose,
    ScaledDense,
)
from networks.wgan import WGAN
from utils import logger
from utils.image_operations import generate_images

#  TODO Add genre information
#  TODO Add artist name
#  TODO add album name


class ProGANDiscriminatorFinalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_year_information: bool = False,
    ):
        """
        Final block for the Discriminator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            add_year_information: Flag to add the year information of the cover
        """
        super(ProGANDiscriminatorFinalBlock, self).__init__()
        final_block_in_channels = in_channels + 1
        self.add_year_information = add_year_information
        if self.add_year_information:
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

    def forward(
        self, images: torch.Tensor, year: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            images: Input tensor of images
            year: Input tensor containing the release year of the images
        """
        x = MinibatchStdDev()(images)
        if self.add_year_information:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        add_year_information: bool = False,
    ):
        """
        Builds the ProGAN discriminator.

        Args:
            n_blocks: Number of blocks.
            n_channels: Number of input channels
            latent_size: Latent size of the corresponding generator
            add_year_information: Flag to take year information into
                consideration during discrimination

        References:
            - Progressive Growing of GANs for Improved Quality, Stability, and Variation: https://arxiv.org/abs/1710.10196
        """
        super().__init__()
        self.add_year_information = add_year_information
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
                add_year_information=add_year_information,
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
        images: torch.Tensor,
        year: torch.Tensor = None,
        block: int = 0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass of the ProGAN discriminator.

        Args:
            x: Input tensor
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        add_year_information: bool = False,
    ):
        """
        Generator Model of the ProGAN network.

        Args:
            n_blocks: Depth of the network
            n_channels: Number of output channels (default = 3 for RGB)
            latent_size: Latent space dimensions
        """
        super().__init__()
        self.add_year_information = add_year_information
        if add_year_information:
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
        self, x: torch.Tensor, block: int, alpha: float
    ) -> torch.Tensor:
        """
        Forward pass of the Generator.

        Args:
            x: input latent noise
            block: depth from where the network's output is required
            alpha: value of alpha for fade-in effect
        """

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
        return {
            "conf": {
                "depth": self.n_blocks,
                "num_channels": self.n_channels,
                "latent_size": self.latent_size,
                "use_eql": self.use_eql,
            },
            "state_dict": self.state_dict(),
        }


class ProGAN(WGAN):
    def __init__(
        self,
        gradient_penalty_weight: float,
        img_height: int,
        img_width: int,
        channels: int = 3,
        latent_size: int = 128,
        use_gpu: bool = False,
        n_blocks: int = 7,
        **kwargs,
    ):
        """
        Progressive growing GAN.

        Progressive Growing GAN that iteratively adds convolutional blocks to
        generator and discriminator. This results in improved quality,
        stability, variation and a faster learning time. Initialization itself
        is similar to the a normal Wasserstein GAN. Training, however, is
        slightly different: Initialization of the training phase should be done
        on a small scale image (e.g. 4x4). After an initial burn-in phase
        (train the GAN on the current resolution), a fade-in phase follows;
        meaning that on fades in the 8x8 upscale layer using the add
        add_fade_in_layers method. This phase is followed by another burn-in
        phase on the current image resolution. The procedure is repeated until
        the output image has the desired resolution.

        Args:
            gradient_penalty_weight: weight for the gradient penalty
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the
                image
        """
        self.n_blocks = n_blocks
        self.latent_size = latent_size
        super(ProGAN, self).__init__(
            img_height=img_height,
            img_width=img_width,
            channels=channels,
            latent_size=latent_size,
            gradient_penalty_weight=gradient_penalty_weight,
            use_gpu=use_gpu,
            **kwargs,
        )
        self.block_images_shown = {
            "burn_in": defaultlist(int),
            "fade_in": defaultlist(int),
        }
        self.add_year_information = kwargs.get("add_year_information", False)
        self.release_year_scaler = None

    def build_discriminator(self, **kwargs) -> ProGANDiscriminator:
        """
        Builds the ProGAN Discriminator.
        """
        return ProGANDiscriminator(
            n_blocks=self.n_blocks,
            n_channels=self.channels,
            latent_size=self.latent_size,
            add_year_information=kwargs.get("add_year_information", False),
        )

    def build_generator(self, **kwargs) -> ProGANGenerator:
        """
        Builds the ProGAN Generator.
        """
        return ProGANGenerator(
            n_blocks=self.n_blocks,
            n_channels=self.channels,
            latent_size=self.latent_size,
            add_year_information=kwargs.get("add_year_information", False),
        )

    def train(
        self,
        data_loader,
        block: int,
        global_steps: int,
        batch_size: int,
        **kwargs,
    ):
        """
        Trains the Progressive growing GAN.

        Args:
            data_loader: Data Loader used for training
            block: Block to train
            global_steps: Absolute number of training steps
            batch_size: Batch size for training

        Keyword Args:
            path: Path to which model training graphs will be written
            write_model_to: Path that can be passed to write the model to during
                training
            grad_acc_steps: Gradient accumulation steps. Ideally a factor of the
                batch size. Otherwise not the entire batch will be used for
                training
        """
        path = kwargs.get("path", ".")
        model_dump_path = kwargs.get("write_model_to", None)
        steps = global_steps // batch_size
        alphas = np.linspace(0, 1, steps).tolist()

        fade_images_shown = self.block_images_shown.get("fade_in")[block]

        if block == 0 or (fade_images_shown // batch_size) == steps:
            phase = "burn_in"
            logger.info(f"Starting burn in for resolution {2 ** (block + 2)}")
            phase_images_shown = self.block_images_shown.get("burn_in")[block]
        else:
            phase = "fade_in"
            logger.info(f"Starting fade in for resolution {2 ** (block + 2)}")
            phase_images_shown = self.block_images_shown.get("fade_in")[block]
            for _ in range(phase_images_shown // batch_size):
                alphas.pop(0)

        for step in range(phase_images_shown // batch_size, steps):
            batch = data_loader.__getitem__(step)
            alpha = alphas.pop(0) if phase == "fade_in" else 1.0

            self.train_on_batch(
                batch,
                alpha=alpha,
                n_critic=kwargs.get("n_critic", 1),
                block=block,
            )
            self.images_shown += batch_size
            if phase == "fade_in":
                self.block_images_shown["fade_in"][block] += batch_size
            else:
                self.block_images_shown["burn_in"][block] += batch_size
            if step % (steps // 32) == 0:
                self._print_output()

                for s in range(25):
                    img_path = os.path.join(
                        path, f"{s}_fixed_step_gif{self.images_shown}.png"
                    )
                    generate_images(
                        self.generator,
                        img_path,
                        target_size=(256, 256),
                        seed=s,
                        n_imgs=1,
                        block=block,
                        alpha=alpha,
                        use_gpu=self.use_gpu,
                        release_year_scaler=self.release_year_scaler,
                    )

                for _k, v in self.metrics.items():
                    plot_metric(
                        path,
                        steps=self.images_shown,
                        metric=v.get("values"),
                        y_label=v.get("label"),
                        file_name=v.get("file_name"),
                    )
                if model_dump_path:
                    self.save(model_dump_path)
        self.save(model_dump_path)
        if phase == "fade_in":
            self.train(
                data_loader=data_loader,
                block=block,
                global_steps=global_steps,
                batch_size=batch_size,
                minibatch_reps=1,
                path=path,
                write_model_to=model_dump_path,
            )

    def train_discriminator(
        self, batch: Dict[str, torch.Tensor], noise: torch.Tensor, **kwargs
    ):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Real input images used for training
            noise: Noise to use from image generation

        Returns:
            the losses for this training iteration
        """
        block = kwargs.get("block")
        alpha = kwargs.get("alpha")

        if self.use_gpu:
            batch = {k: v.cuda() for k, v in batch.items()}
            noise = noise.cuda()

        if self.add_year_information:
            noise = torch.cat((noise, batch.get("year")), 1)
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with torch.no_grad():
            fake_images = self.generator(noise, block=block, alpha=alpha)

        fake_pred = self.discriminator(
            images=fake_images,
            year=batch.get("year"),
            block=block,
            alpha=alpha,
        )
        real_pred = self.discriminator(
            images=batch.get("images"),
            year=batch.get("year"),
            block=block,
            alpha=alpha,
        )
        loss = torch.mean(fake_pred) - torch.mean(real_pred)
        fake_batch = {"images": fake_images, "year": batch.get("year")}
        gp = self._gradient_penalty(
            batch, fake_batch, block=block, alpha=alpha
        )
        loss += gp
        loss += 0.001 * torch.mean(real_pred ** 2)
        loss.backward()
        self.discriminator_optimizer.step()
        return loss

    def train_generator(
        self, batch: Dict[str, torch.Tensor], noise: torch.Tensor, **kwargs
    ):
        block = kwargs.get("block")
        alpha = kwargs.get("alpha")

        if self.use_gpu:
            batch = {k: v.cuda() for k, v in batch.items()}
            noise = noise.cuda()

        if self.add_year_information:
            noise = torch.cat((noise, batch.get("year")), 1)

        self.generator.train()
        self.generator_optimizer.zero_grad()
        generated_images = self.generator(noise, block=block, alpha=alpha)
        fake_pred = self.discriminator(
            images=generated_images,
            year=batch.get("year"),
            block=block,
            alpha=alpha,
        )
        loss_fake = -torch.mean(fake_pred)
        loss_fake.backward()
        self.generator_optimizer.step()
        return loss_fake.detach().cpu().numpy().tolist()

    def load(self, path):
        gan = super(ProGAN, self).load(path)
        self.block_images_shown = gan.block_images_shown
