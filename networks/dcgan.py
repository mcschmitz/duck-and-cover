import os
from typing import Tuple

import numpy as np
import torch
from torch import nn

from networks.gan import GAN
from networks.utils import clip_channels, plot_metric
from utils import logger
from utils.image_operations import generate_images


class DCDiscrimininator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int], use_gpu: bool = False):
        """
        Builds the DCGAN discriminator.

        Builds the very simple discriminator that takes an image input and
        applies a 3x3 convolutional layer with ReLu activation and a 2x2 stride
        until the desired embedding  size is reached. The flattened embedding is
        ran through a Dense layer with sigmoid output to label the image.

        Args:
            img_shape: Shape of the input image
            use_gpu: Flag to train on GPU
        """
        super(DCDiscrimininator, self).__init__()
        self.use_gpu = use_gpu
        self.img_shape = img_shape
        n_filters = clip_channels(16)
        self.init_conv2d = nn.Conv2d(
            img_shape[0], n_filters, kernel_size=3, stride=2, padding=1
        )
        nn.init.kaiming_normal_(self.init_conv2d.weight)
        cur_img_size = self.img_shape[2] // 2
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
            self.conv2d_layers[-1].out_channels * cur_img_size ** 2
        )
        self.final_linear = nn.Linear(final_linear_input_dim, 1)
        nn.init.xavier_uniform_(self.final_linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method of the discriminator.

        Args:
            x: Tensor to pass through the model
        """
        if self.use_gpu:
            x = x.cuda()
        x = self.init_conv2d(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = nn.Dropout(0.3)(x)
        for conv2d_layer in self.conv2d_layers:
            x = conv2d_layer(x)
            x = nn.LeakyReLU(negative_slope=0.3)(x)
            x = nn.Dropout(0.3)(x)

        x = nn.Flatten()(x)
        x = self.final_linear(x)
        return nn.Sigmoid()(x)


class DCGenerator(nn.Module):
    def __init__(
        self,
        latent_size: int,
        img_shape: Tuple[int, int, int],
        use_gpu: bool = False,
    ):
        """
        Builds the DCGAN generator.

        Builds the very simple generator that takes a latent input vector and
        applies the following block until the desired image size is reached:
        3x3 convolutional layer with ReLu activation -> Batch Normalization
        -> Upsamling layer. The last Convolutional layer wit tanH activation
        results in 3 RGB channels and serves as final output

        Args:
            latent_size: Dimensions of the latent vector
            img_shape: Shape of the input image
            use_gpu: Flag to train on GPU
        """
        super(DCGenerator, self).__init__()
        self.use_gpu = use_gpu
        self.img_shape = img_shape
        self.latent_size = latent_size
        n_filters = clip_channels(self.latent_size)
        self.initial_linear = nn.Linear(self.latent_size, n_filters * 4 * 4)
        nn.init.xavier_uniform_(self.initial_linear.weight)
        self.init_batch_norm = nn.BatchNorm1d(self.initial_linear.out_features)
        self.init_conv2d = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=1,
            stride=1,
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
        Forward pass of the Deep Convolutional Discriminator.

        Args:
            x: Input Tensor
        """
        if self.use_gpu:
            x = x.cuda()
        x = self.initial_linear(x)
        x = self.init_batch_norm(x)
        x = nn.LeakyReLU(negative_slope=0.3)(x)
        x = torch.reshape(x, (-1, self.latent_size, 4, 4))
        x = self.init_conv2d(x)

        for conv2d_layer, batch_norm_layer in zip(
            self.conv2d_layers, self.batch_norm_layers
        ):
            x = nn.UpsamplingNearest2d(scale_factor=2)(x)
            x = conv2d_layer(x)
            x = batch_norm_layer(x)
            x = nn.LeakyReLU(negative_slope=0.3)(x)

        x = self.final_conv2d(x)
        return nn.Tanh()(x)


class DCGAN(GAN):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        channels: int = 3,
        latent_size: int = 128,
        use_gpu: bool = False,
    ):
        """
        Simple Deep Convolutional GAN.

        Simple DCGAN that builds a discriminator a generator and the adversarial
        model to train the GAN based on the binary crossentropy loss for the
        generator and the discriminator.

        Args:
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the
                image
            use_gpu: Flag to use the GPU for training
        """
        super(DCGAN, self).__init__(use_gpu)
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.channels, self.img_height, self.img_width)
        self.latent_size = latent_size
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        if self.use_gpu:
            self.discriminator.cuda()
            self.generator.cuda()

        self.metrics["D_accuracy"] = {
            "file_name": "d_acc.png",
            "label": "Discriminator Accuracy",
            "values": [],
        }
        self.metrics["G_loss"] = {
            "file_name": "g_loss.png",
            "label": "Generator Loss",
            "values": [],
        }

    def build_generator(self) -> DCGenerator:
        """
        Builds the class specific generator.
        """
        return DCGenerator(self.latent_size, self.img_shape, self.use_gpu)

    def build_discriminator(self) -> DCDiscrimininator:
        """
        Builds the class specific discriminator.
        """
        return DCDiscrimininator(self.img_shape, self.use_gpu)

    def train_on_batch(self, real_images: torch.Tensor, **kwargs):
        """
        Runs a single gradient update on a batch of data.

        Args:
            real_images: numpy array of real input images used for training
        """
        noise = torch.normal(
            mean=0, std=1, size=(len(real_images), self.latent_size)
        )
        self.metrics["D_accuracy"]["values"].append(
            self.train_discriminator(real_images, noise)
        )
        self.metrics["G_loss"]["values"].append(self.train_generator(noise))

    def train_discriminator(
        self, real_images: torch.Tensor, noise: torch.Tensor
    ) -> float:
        """
        Runs a single gradient update for the discriminator.

        Args:
            real_images: Array of real images
            noise: Random noise input Tensor
        """
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with torch.no_grad():
            generated_images = self.generator(noise)
        fake = torch.ones((len(generated_images), 1))
        real = torch.zeros((len(real_images), 1))
        if self.use_gpu:
            fake = fake.cuda()
            real = real.cuda()
        fake_pred = self.discriminator(generated_images)
        real_pred = self.discriminator(real_images)
        loss_fake = nn.BCELoss()(fake_pred, fake)
        loss_real = nn.BCELoss()(real_pred, real)
        loss = (loss_fake + loss_real) / 2
        loss.backward()
        self.discriminator_optimizer.step()
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        return (correct_real + correct_fake) / (len(generated_images) * 2)

    def train_generator(self, noise: torch.Tensor) -> float:
        """
        Runs a single gradient update for the generator.

        Args:
            noise: Random noise input Tensor
        """
        self.generator.train()
        self.generator_optimizer.zero_grad()
        fake = torch.ones((len(noise), 1))
        if self.use_gpu:
            fake = fake.cuda()
        generated_images = self.generator(noise)
        fake_pred = self.discriminator(generated_images)
        loss_fake = nn.BCELoss()(fake_pred, fake)
        loss_fake.backward()
        self.generator_optimizer.step()
        return loss_fake.detach().cpu().numpy().tolist()

    def train(self, data_loader, global_steps, batch_size, **kwargs):
        """
        Trains the network.

        Args:
            data_loader: Data Loader to stream training batches to the network
            global_steps: Absolute numbers of steps to train
            batch_size: Size of the trainin batches
            **kwargs: Keyword arguments. Add `path` to change the output path
        """
        path = kwargs.get("path", ".")
        model_dump_path = kwargs.get("write_model_to", None)
        steps = global_steps // batch_size
        for step in range(self.images_shown // batch_size, steps):
            batch = data_loader.__getitem__(step)
            self.train_on_batch(batch, **kwargs)
            self.images_shown += batch_size

            if step % (steps // 320) == 0:
                self._print_output()
                self._generate_images(path)

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
        if model_dump_path:
            self.save(model_dump_path)

    def _generate_images(self, path):
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
            )

    def _print_output(self):
        g_loss = np.round(
            np.mean(self.metrics["G_loss"]["values"]), decimals=3
        )
        d_acc = np.round(
            np.mean(self.metrics["D_accuracy"]["values"]), decimals=3
        )
        logger.info(
            f"Images shown {self.images_shown}:"
            + f" Generator Loss: {g_loss} -"
            + f" Discriminator Acc.: {d_acc}"
        )
