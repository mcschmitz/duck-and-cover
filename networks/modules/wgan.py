from typing import Tuple

import torch
from torch import Tensor, nn

from networks.utils import clip_channels
from networks.utils.layers import MinibatchStdDev


class WGANDiscriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int]):
        """
        Builds the WGAN discriminator.

        Builds the very simple discriminator that takes an image input and
        applies a 3x3 convolutional layer with ReLu activation and a 2x2 stride
        until the desired embedding  size is reached. The flattened embedding is
        ran through a Dense layer with sigmoid output to label the image.

        Args:
            img_shape: Shape of the input image
        """
        super().__init__()
        self.img_shape = img_shape
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
            (self.conv2d_layers[-1].out_channels + 1) * cur_img_size**2
        )
        self.final_linear = nn.Linear(final_linear_input_dim, 1)
        nn.init.kaiming_normal_(self.final_linear.weight)

    def forward(self, images: Tensor, **kwargs) -> Tensor:
        """
        Forward method of the discriminator.

        Args:
            images: Tensor to pass through the model
            kwargs: Keyword Arguments that can be used by child classes
        """
        images = self.init_conv2d(images)
        for conv2d_layer in self.conv2d_layers:
            images = conv2d_layer(images)
            images = nn.LeakyReLU(negative_slope=0.3)(images)

        images = MinibatchStdDev()(images)
        images = nn.Flatten()(images)
        return self.final_linear(images)


class WGANGenerator(nn.Module):
    def __init__(
        self,
        latent_size: int,
        img_shape: Tuple[int, int, int],
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
        """
        super().__init__()
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the generator.

        Args:
            x: Input Tensor
        """
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
