from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from networks.dcgan import DCGAN, DCDiscrimininator, DCGenerator
from networks.utils import clip_channels
from networks.utils.layers import MinibatchStdDev
from utils import logger


class WGANDiscriminator(DCDiscrimininator):
    def __init__(self, img_shape: Tuple[int, int, int], use_gpu: bool = False):
        """
        Builds the WGAN discriminator.

        Builds the very simple discriminator that takes an image input and
        applies a 3x3 convolutional layer with ReLu activation and a 2x2 stride
        until the desired embedding  size is reached. The flattened embedding is
        ran through a Dense layer with sigmoid output to label the image.

        Args:
            img_shape: Shape of the input image
            use_gpu: Flag to train on GPU
        """
        super().__init__(img_shape=img_shape, use_gpu=use_gpu)
        n_filters = clip_channels(16)
        self.init_conv2d = nn.Conv2d(
            img_shape[0],
            n_filters,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        nn.init.kaiming_normal_(self.init_conv2d.weight)
        cur_img_size = self.img_shape[2]
        self.conv2d_layers = nn.ModuleList()
        while cur_img_size > 4:
            conv2d_layer = nn.Conv2d(
                clip_channels(n_filters),
                clip_channels(n_filters * 2),
                kernel_size=3,
                stride=2,
                padding=1,
            )
            nn.init.kaiming_normal_(conv2d_layer.weight)
            self.conv2d_layers.append(conv2d_layer)
            n_filters *= 2
            cur_img_size /= 2

        final_linear_input_dim = int(
            self.conv2d_layers[-1].out_channels
            * cur_img_size
            * (cur_img_size + 1)
        )
        self.final_linear = nn.Linear(final_linear_input_dim, 1)
        nn.init.kaiming_normal_(self.final_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the discriminator.

        Args:
            x: Tensor to pass through the model
        """
        if self.use_gpu:
            x = x.cuda()
        x = self.init_conv2d(x)
        for conv2d_layer in self.conv2d_layers:
            x = conv2d_layer(x)
            x = nn.LeakyReLU(negative_slope=0.3)(x)

        x = MinibatchStdDev()(x)
        x = nn.Flatten()(x)
        return self.final_linear(x)


class WGANGenerator(DCGenerator):
    def __init__(
        self,
        latent_size: int,
        img_shape: Tuple[int, int, int],
        use_gpu: bool = False,
    ):
        """
        Builds the WGAN generator.

        Builds the very simple generator that takes a latent input vector and
        applies the following block until the desired image size is reached:
        3x3 convolutional layer with ReLu activation -> Batch Normalization
        -> Upsampling layer. The last Convolutional layer wit tanH activation
        results in 3 RGB channels and serves as final output

        Args:
            latent_size: Dimensions of the latent vector
            img_shape: Shape of the input image
            use_gpu: Flag to train on GPU
        """
        super().__init__(
            latent_size=latent_size, use_gpu=use_gpu, img_shape=img_shape
        )
        n_filters = clip_channels(self.latent_size)
        self.initial_linear = nn.Linear(self.latent_size, n_filters * 4 * 4)
        nn.init.xavier_uniform_(self.initial_linear.weight)
        self.init_batch_norm = nn.BatchNorm1d(self.initial_linear.out_features)
        self.init_conv2d = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        nn.init.kaiming_normal_(self.init_conv2d.weight)
        cur_img_size = 4
        self.conv2d_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        while cur_img_size < self.img_shape[2]:
            conv2d_layer = nn.Conv2d(
                clip_channels(n_filters),
                clip_channels(n_filters // 2),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            nn.init.kaiming_normal_(conv2d_layer.weight)
            cur_img_size *= 2
            n_filters = clip_channels(n_filters // 2)
            self.conv2d_layers.append(conv2d_layer)
            self.batch_norm_layers.append(nn.BatchNorm2d(n_filters))

        self.final_conv2d = nn.Conv2d(
            n_filters, self.img_shape[0], kernel_size=1, stride=1
        )
        nn.init.xavier_uniform_(self.final_conv2d.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x: Input Tensor
        """
        if self.use_gpu:
            x = x.cuda()
        x = self.initial_linear(x)
        x = nn.LeakyReLU(negative_slope=0.2)(x)
        x = self.init_batch_norm(x)
        x = torch.reshape(x, (-1, self.latent_size, 4, 4))
        x = self.init_conv2d(x)

        for conv2d_layer, batch_norm_layer in zip(
            self.conv2d_layers, self.batch_norm_layers
        ):
            x = nn.UpsamplingNearest2d(scale_factor=2)(x)
            x = conv2d_layer(x)
            x = nn.LeakyReLU(negative_slope=0.2)(x)
            x = batch_norm_layer(x)

        x = self.final_conv2d(x)
        return nn.Tanh()(x)


class WGAN(DCGAN):
    def __init__(
        self,
        gradient_penalty_weight: float,
        img_height: int,
        img_width: int,
        channels: int = 3,
        latent_size: int = 128,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        Wasserstein GAN.

        Wasserstein GAN with gradient penalty, that builds a discriminator a
        generator and the adversarial model to train the GAN based on the
        wasserstein loss for negative, positive and interpolated examples

        Args:
            gradient_penalty_weight: weight for the gradient penalty
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the
                image
            use_gpu: Flag to use the GPU for training and prediction
        """
        super(WGAN, self).__init__(
            img_height=img_height,
            img_width=img_width,
            channels=channels,
            latent_size=latent_size,
            use_gpu=use_gpu,
            **kwargs,
        )
        self._gradient_penalty_weight = gradient_penalty_weight
        self.metrics["D_loss"] = {
            "file_name": "d_loss.png",
            "label": "Discriminator Loss",
            "values": [],
        }

    def build_generator(self) -> WGANGenerator:
        """
        Builds the WGAN Generator.
        """
        return WGANGenerator(self.latent_size, self.img_shape, self.use_gpu)

    def build_discriminator(self) -> WGANDiscriminator:
        """
        Builds the WGAN Discriminator.
        """
        return WGANDiscriminator(self.img_shape, self.use_gpu)

    def train_on_batch(self, batch: np.ndarray, **kwargs):
        """
        Runs a single gradient update on a batch of data.

        Args:
        """
        n_critic = kwargs.get("n_critic", 5)

        d_accuracies = []
        for _ in range(n_critic):
            noise = torch.normal(
                mean=0, std=1, size=(len(batch["images"]), self.latent_size)
            )
            d_accuracies.append(
                self.train_discriminator(batch, noise, **kwargs)
            )
        d_accuracies = np.mean([d.detach().tolist() for d in d_accuracies])
        self.metrics["D_loss"]["values"].append(np.mean(d_accuracies))

        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.latent_size)
        )
        self.metrics["G_loss"]["values"].append(
            self.train_generator(batch, noise, **kwargs)
        )

    def train_discriminator(self, batch: torch.Tensor, noise: torch.Tensor):
        """
        Runs a single gradient update on a batch of data.

        Args:
            noise: Noise to use from image generation

        Returns:
            the losses for this training iteration
        """
        if self.use_gpu:
            batch = batch.cuda()
            noise = noise.cuda()

        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with torch.no_grad():
            generated_images = self.generator(noise)
        fake_pred = self.discriminator(generated_images)
        real_pred = self.discriminator(batch)
        loss = torch.mean(fake_pred) - torch.mean(real_pred)
        gp = self._gradient_penalty(batch, generated_images)
        loss += gp
        loss.backward()
        self.discriminator_optimizer.step()
        return loss

    def train_generator(self, noise: torch.Tensor):
        self.generator.train()
        self.generator_optimizer.zero_grad()
        generated_images = self.generator(noise)
        fake_pred = self.discriminator(generated_images)
        loss_fake = -torch.mean(fake_pred)
        loss_fake.backward()
        self.generator_optimizer.step()
        return loss_fake.detach().cpu().numpy().tolist()

    def _gradient_penalty(
        self,
        real_batch: Dict[str, torch.Tensor],
        fake_batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        batch_size = real_batch["images"].shape[0]

        random_avg_weights = torch.rand((batch_size, 1, 1, 1)).to(
            real_batch["images"].device
        )
        random_avg = random_avg_weights * real_batch["images"] + (
            (1 - random_avg_weights) * fake_batch["images"]
        )
        random_avg.requires_grad_(True)

        pred = self.discriminator(
            images=random_avg, year=real_batch["year"], **kwargs
        )
        grad = torch.autograd.grad(
            outputs=pred,
            inputs=random_avg,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grad.view(grad.shape[0], -1)
        return (
            self._gradient_penalty_weight
            * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
        )

    def _print_output(self):
        g_loss = np.round(
            np.mean(self.metrics["G_loss"]["values"]), decimals=3
        )
        d_loss = np.round(
            np.mean(self.metrics["D_loss"]["values"]), decimals=3
        )
        logger.info(
            f"Images shown {self.images_shown}:"
            + f" Generator Loss: {g_loss} -"
            + f" Discriminator Loss: {d_loss}"
        )
