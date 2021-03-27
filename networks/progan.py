import copy
import os
from typing import Any, Dict

import joblib
import numpy as np
import torch
from defaultlist import defaultlist
from tensorflow.keras import Model
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

#  TODO Add release year information
#  TODO Add genre information
#  TODO Add artist name
#  TODO add album name


class ProGANDiscriminatorFinalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Final block for the Discriminator

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(ProGANDiscriminatorFinalBlock, self).__init__()
        self.conv_1 = ScaledConv2d(
            in_channels + 1,
            in_channels,
            (3, 3),
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            in_channels,
            out_channels,
            (4, 4),
            bias=True,
            use_dynamic_wscale=True,
        )
        final_linear_input_dim = int(self.conv_2.out_channels)
        self.final_linear = ScaledDense(final_linear_input_dim, 1, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the module

        Args:
            x: Input tensor
        """
        x = MinibatchStdDev()(x)
        x = nn.LeakyReLU(0.2)(self.conv_1(x))
        x = nn.LeakyReLU(0.2)(self.conv_2(x))
        return self.final_linear(x)


class DisGeneralConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        General block used in the discriminator

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super(DisGeneralConvBlock, self).__init__()

        self.conv_1 = ScaledConv2d(
            in_channels,
            in_channels,
            (3, 3),
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            in_channels,
            out_channels,
            (3, 3),
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the general block

        Args:
            x: Input tensor
        """
        x = nn.LeakyReLU(0.2)(self.conv_1(x))
        x = nn.LeakyReLU(0.2)(self.conv_2(x))
        return nn.AvgPool2d(2)(x)


class ProGANDiscriminator(nn.Module):
    def __init__(
        self,
        n_blocks: int = 7,
        n_channels: int = 3,
        latent_size: int = 512,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.num_channels = n_channels
        self.latent_size = latent_size

        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                ScaledConv2d(
                    n_channels,
                    calc_channels_at_stage(n_blocks),
                    kernel_size=(1, 1),
                    use_dynamic_wscale=True,
                ),
                nn.LeakyReLU(0.2),
            )
        )

        for block in reversed(range(1, n_blocks)):
            self.layers.append(
                DisGeneralConvBlock(
                    calc_channels_at_stage(block + 1),
                    calc_channels_at_stage(block),
                ),
            )

        self.layers.append(
            ProGANDiscriminatorFinalBlock(
                calc_channels_at_stage(1), latent_size // 2
            )
        )

    def forward(
        self, x: torch.Tensor, depth: int, alpha: float
    ) -> torch.Tensor:
        if depth > self.n_blocks:
            raise ValueError(
                f"This model only has {self.n_blocks} blocks. depth parameter has to be <= n_blocks"
            )

        if depth > 2:
            residual = self.from_rgb[-(depth - 2)](
                nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            )
            straight = self.layers[-(depth - 1)](
                self.from_rgb[-(depth - 1)](x)
            )
            y = (alpha * straight) + ((1 - alpha) * residual)
            for layer_block in self.layers[-(depth - 2) : -1]:
                y = layer_block(y)
        else:
            y = self.from_rgb[-1](x)
        y = self.layers[-1](y)
        return y

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "depth": self.n_blocks,
                "num_channels": self.num_channels,
                "latent_size": self.latent_size,
                "use_eql": self.use_eql,
                "num_classes": self.num_classes,
            },
            "state_dict": self.state_dict(),
        }


class GenInitialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Module implementing the initial block of the input
        Args:
            in_channels: number of input channels to the block
            out_channels: number of output channels of the block
        """
        super(GenInitialBlock, self).__init__()

        self.conv_1 = ScaledConv2dTranspose(
            in_channels,
            out_channels,
            (4, 4),
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            out_channels,
            out_channels,
            (3, 3),
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        y = PixelwiseNorm()(y)  # normalize the latents to hypersphere
        y = nn.LeakyReLU(0.2)(self.conv_1(y))
        y = nn.LeakyReLU(0.2)(self.conv_2(y))
        y = PixelwiseNorm()(y)
        return y


class GenGeneralConvBlock(nn.Module):
    """
    Module implementing a general convolutional block
    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels required
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(GenGeneralConvBlock, self).__init__()

        self.conv_1 = ScaledConv2d(
            in_channels,
            out_channels,
            (3, 3),
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )
        self.conv_2 = ScaledConv2d(
            out_channels,
            out_channels,
            (3, 3),
            padding=1,
            bias=True,
            use_dynamic_wscale=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.interpolate(x, scale_factor=2)
        y = PixelwiseNorm()(nn.LeakyReLU(0.2)(self.conv_1(y)))
        y = PixelwiseNorm()(nn.LeakyReLU(0.2)(self.conv_2(y)))

        return y


class ProGANGenerator(nn.Module):
    def __init__(
        self,
        n_blocks: int = 10,
        n_channels: int = 3,
        latent_size: int = 512,
    ):
        """
        Generator Module (block) of the GAN network
        Args:
            n_blocks: required depth of the Network
            n_channels: number of output channels (default = 3 for RGB)
            latent_size: size of the latent manifold
        """
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

        for block in range(1, n_blocks):
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

    def forward(
        self, x: torch.Tensor, depth: int, alpha: float
    ) -> torch.Tensor:
        """
        forward pass of the Generator
        Args:
            x: input latent noise
            depth: depth from where the network's output is required
            alpha: value of alpha for fade-in effect
        Returns: generated images at the give depth's resolution
        """

        assert (
            depth <= self.n_blocks
        ), f"Requested output depth {depth} cannot be produced"

        if depth == 2:
            y = self.rgb_converters[0](self.layers[0](x))
        else:
            y = x
            for layer_block in self.layers[: depth - 2]:
                y = layer_block(y)
            residual = nn.functional.interpolate(
                self.rgb_converters[depth - 3](y), scale_factor=2
            )
            straight = self.rgb_converters[depth - 2](
                self.layers[depth - 2](y)
            )
            y = (alpha * straight) + ((1 - alpha) * residual)
        return y

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
        )
        self.block_images_shown = {
            "burn_in": defaultlist(int),
            "fade_in": defaultlist(int),
        }

    def build_discriminator(self) -> ProGANDiscriminator:
        """
        Builds the ProGAN Discriminator.
        """
        return ProGANDiscriminator(
            n_blocks=self.n_blocks,
            n_channels=self.channels,
            latent_size=self.latent_size,
        )

    def build_generator(self) -> ProGANGenerator:
        """
        Builds the ProGAN Generator.
        """
        return ProGANGenerator(
            n_blocks=self.n_blocks,
            n_channels=self.channels,
            latent_size=self.latent_size,
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
            verbose: Boolean switch for verbosity
            write_model_to: Path that can be passed to write the model to during
                training
            grad_acc_steps: Gradient accumulation steps. Ideally a factor of the
                batch size. Otherwise not the entire batch will be used for
                training
        """
        path = kwargs.get("path", ".")
        steps = global_steps // batch_size
        verbose = kwargs.get("verbose", True)
        model_dump_path = kwargs.get("write_model_to", None)
        n_critic = kwargs.get("n_critic", 1)

        fade_images_shown = self.block_images_shown.get("fade_in")[block]
        if block == 0 or (fade_images_shown // batch_size) == steps:
            phase = "burn_in"
            logger.info(f"Starting burn in for resolution {2 ** (block + 2)}")
            phase_images_shown = self.block_images_shown.get("burn_in")[block]
        else:
            phase = "fade_in"
            logger.info(f"Starting fade in for resolution {2 ** (block + 2)}")
            phase_images_shown = self.block_images_shown.get("fade_in")[block]

        alphas = np.linspace(0, 1, steps).tolist()
        for step in range(phase_images_shown // batch_size, steps):
            batch = data_loader.__getitem__(step)
            if phase == "fade_in":
                self._update_alpha(alphas.pop(0))

            self.train_on_batch(batch, n_critic=n_critic, phase=phase)
            self.images_shown += batch_size
            if phase == "fade_in":
                self.block_images_shown["fade_in"][block] += batch_size
            else:
                self.block_images_shown["burn_in"][block] += batch_size
            if step % (steps // 32) == 0 and verbose:
                self._print_output()
                self._generate_images(path, phase)

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
                verbose=True,
                minibatch_reps=1,
                n_critic=n_critic,
                path=path,
                write_model_to=model_dump_path,
            )

    def _generate_images(self, path, phase):
        suffix = "_fade_in" if phase == "fade_in" else ""
        for s in range(25):
            img_path = os.path.join(
                path, f"{s}_fixed_step_gif{self.images_shown}.png"
            )
            generate_images(
                getattr(self, f"generator{suffix}"),
                img_path,
                target_size=(256, 256),
                seed=s,
                n_imgs=1,
            )

        if phase == "fade_in":
            min_max_alpha = (0, 1)
            for a in min_max_alpha:
                self._update_alpha(a)
                img_path = os.path.join(
                    path,
                    f"fixed_step{self.images_shown}_alpha{a}.png",
                )
                generate_images(
                    getattr(self, f"generator{suffix}"),
                    img_path,
                    target_size=(256, 256),
                    seed=101,
                )

    def save(self, path):
        gan = copy.copy(self)
        fade_in_img_shown = self.block_images_shown["fade_in"]
        burn_in_img_shown = self.block_images_shown["burn_in"]
        if fade_in_img_shown[-1] < burn_in_img_shown[-1] and len(
            burn_in_img_shown
        ) < len(fade_in_img_shown):
            model_to_save = getattr(gan, "generator_fade_in")
            if model_to_save:
                model_to_save.save_weights(os.path.join(path, "G_fade_in.h5"))
                getattr(gan, f"discriminator_fade_in").save_weights(
                    os.path.join(path, f"D_fade_in.h5")
                )
                getattr(gan, f"discriminator_model_fade_in").save_weights(
                    os.path.join(path, f"DM_fade_in.h5")
                )
                getattr(gan, f"combined_model_fade_in").save_weights(
                    os.path.join(path, f"C_fade_in.h5")
                )
        else:
            model_to_save = getattr(gan, "generator")
            if model_to_save:
                model_to_save.save_weights(os.path.join(path, "G.h5"))
                getattr(gan, f"discriminator").save_weights(
                    os.path.join(path, f"D.h5")
                )
                getattr(gan, f"discriminator_model").save_weights(
                    os.path.join(path, f"DM.h5")
                )
                getattr(gan, f"combined_model").save_weights(
                    os.path.join(path, f"C.h5")
                )
        for k, v in gan.__dict__.items():
            if isinstance(v, Model):
                setattr(gan, k, None)
        joblib.dump(gan, os.path.join(path, "GAN.pkl"))

    def load(self, path):
        models = ["_fade_in"]
        gan = joblib.load(os.path.join(path, "GAN.pkl"))
        fade_in_img_shown = gan.block_images_shown["fade_in"]
        burn_in_img_shown = gan.block_images_shown["burn_in"]
        if fade_in_img_shown[-1] < burn_in_img_shown[-1] and len(
            burn_in_img_shown
        ) == len(fade_in_img_shown):
            models.append("")
        for fade_in in models:
            weights_to_load = os.path.join(path, f"G{fade_in}.h5")
            if os.path.isfile(weights_to_load):
                getattr(self, f"generator{fade_in}").load(weights_to_load)
                getattr(self, f"discriminator{fade_in}").load(
                    os.path.join(path, f"D{fade_in}.h5")
                )
                getattr(self, f"discriminator_model{fade_in}").load(
                    os.path.join(path, f"DM{fade_in}.h5")
                )
                getattr(self, f"combined_model{fade_in}").load(
                    os.path.join(path, f"C{fade_in}.h5")
                )

        self.images_shown = gan.images_shown
        self.metrics = gan.metrics
        self.block_images_shown = gan.block_images_shown
