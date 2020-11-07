import logging

import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    LeakyReLU,
    Reshape,
    UpSampling2D,
)

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from networks.dcgan import DCGAN
from networks.utils import wasserstein_loss
from networks.utils.layers import (
    GradientPenalty,
    MinibatchSd,
    RandomWeightedAverage,
)

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger(__file__)


class WGAN(DCGAN):
    def __init__(
        self,
        gradient_penalty_weight,
        img_height: int,
        img_width: int,
        channels: int = 3,
        latent_size: int = 128,
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
        """
        super(WGAN, self).__init__(
            img_height=img_height,
            img_width=img_width,
            channels=channels,
            latent_size=latent_size,
        )
        self.discriminator_loss = []
        self._gradient_penalty_weight = gradient_penalty_weight

    def build_models(self, optimizer):
        """
        Builds the desired GAN that allows to generate covers.

        Builds the generator, the discriminator and the combined model for a
        WGAN using Wasserstein loss with gradient penalty to improve learning.

        Args:
            optimizer: Which optimizer to use
        """
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        for layer in self.generator.layers:
            layer.trainable = False
        self.generator.trainable = False
        self._build_discriminator_model(optimizer)
        self.history["D_loss_positives"] = []
        self.history["D_loss_negatives"] = []
        self.history["D_loss_dummies"] = []

        self.fuse_disc_and_gen(optimizer)

    def _build_discriminator(self):
        """
        Defines the architecture for the the WGAN discriminator.

        The discriminator takes an image input and applies a 3x3 convolutional
        layer with LeakyReLu activation and a 2x2 stride until the desired
        embedding  size is reached. The flattend embedding is ran through a
        1024 Dense layer followed by a output layer with one linear output node.

        Returns:
            The WGAN discriminator
        """
        image_input = Input(self.img_shape)
        x = Conv2D(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_initializer="he_normal",
            padding="same",
        )(image_input)

        cur_img_size = self.img_shape[0]
        n_channels = 64
        while cur_img_size > 4:
            x = Conv2D(
                n_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                kernel_initializer="he_normal",
                padding="same",
            )(x)
            x = LeakyReLU()(x)
            n_channels *= 2
            cur_img_size //= 2

        x = MinibatchSd()(x)
        x = Flatten()(x)
        discriminator_output = Dense(1, kernel_initializer="he_normal")(x)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def _build_discriminator_model(self, optimizer):
        """
        Builds the discriminator for the WGAN with gradient penalty.

        The discriminator takes real images, generated ones and an
        average of both and optimizes the wasserstein loss for the real
        and the fake images as well as the gradient penalty for the
        averaged samples

        Args:
            optimizer: Tensorflow optimizer used for training
        """
        disc_input_image = Input(self.img_shape, name="Img_Input")
        disc_input_noise = Input(
            (self.latent_size,), name="Noise_Input_for_Discriminator"
        )
        gen_image_disc = self.generator(disc_input_noise)
        disc_image_gen = self.discriminator(gen_image_disc)
        disc_image_image = self.discriminator(disc_input_image)
        avg_samples = RandomWeightedAverage()(
            [disc_input_image, gen_image_disc]
        )
        disc_avg_disc = self.discriminator(avg_samples)
        gp_layer = GradientPenalty(weight=self._gradient_penalty_weight)
        gp = gp_layer([disc_avg_disc, avg_samples])
        self.discriminator_model = Model(
            inputs=[disc_input_image, disc_input_noise],
            outputs=[disc_image_image, disc_image_gen, gp],
        )
        self.discriminator_model.compile(
            loss=[wasserstein_loss, wasserstein_loss, "mse"],
            optimizer=optimizer,
        )

    def _build_combined_model(self, optimizer):
        """
        Build the combined GAN consisting of generator and discriminator.

        Takes the latent input and generates an images out of it by applying the
        generator. Classifies the image via the discriminator. The model is
        compiled using the given optimizer

        Args:
            optimizer: which optimizer to use
        """
        gen_input_latent = Input((self.latent_size,), name="Latent_Input")
        gen_image = self.generator(gen_input_latent)
        disc_image = self.discriminator(gen_image)
        self.combined_model = Model(gen_input_latent, disc_image)
        self.combined_model.compile(optimizer, loss=[wasserstein_loss])

    def _build_generator(self):
        """
        Defines the architecture for the WGAN.

        The generator takes a latent input vector and applies the following
        block until the desired image size is reached: 3x3 convolutional layer
        -> Upsamling layer -> with ReLu activation -> Batch Normalization. The
        last Convolutional layer wit tanH activation results in 3 RGB channels
        and serves as final output

        Returns:
            The DCGAN generator
        """
        noise_input = Input((self.latent_size,))
        x = Dense(1024)(noise_input)
        x = LeakyReLU()(x)
        x = Dense(4 * 4 * 512)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        x = Reshape((4, 4, 512))(x)

        cur_img_size = 4
        n_channels = 256
        x = Conv2D(
            n_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)
        while cur_img_size < self.img_shape[0]:
            x = UpSampling2D()(x)
            x = Conv2D(
                n_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
                use_bias=False,
            )(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            cur_img_size *= 2
            n_channels //= 2

        generator_output = Conv2D(
            self.channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )(x)
        generator_output = Activation("tanh")(generator_output)
        generator_model = Model(noise_input, generator_output)
        return generator_model

    def train_on_batch(self, real_images, n_critic: int = 5):
        """
        Runs a single gradient update on a batch of data.

        Args:
            real_images: numpy array of real input images used for training
            n_critic: number of discriminator updates for each iteration
        """
        batch_size = real_images.shape[0] // n_critic
        real_y = np.ones((batch_size, 1)) * -1

        for i in range(n_critic):
            discriminator_minibatch = real_images[
                i * batch_size : (i + 1) * batch_size
            ]
            losses = self.train_discriminator(discriminator_minibatch)

        self.history["D_loss_positives"].append(losses[0])
        self.history["D_loss_negatives"].append(losses[1])
        self.history["D_loss_dummies"].append(losses[2])

        noise = np.random.normal(size=(batch_size, self.latent_size))
        self.history["G_loss"].append(
            self.combined_model.train_on_batch(noise, real_y)
        )

    def train_discriminator(self, real_images):
        """
        Runs a single gradient update on a batch of data.

        Args:
            real_images: numpy array of real input images used for training

        Returns:
            the losses for this training iteration
        """
        batch_size = len(real_images)
        fake_y = np.ones((batch_size, 1))
        real_y = np.ones((batch_size, 1)) * -1
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(size=(batch_size, self.latent_size))
        losses = self.discriminator_model.train_on_batch(
            [real_images, noise], [real_y, fake_y, dummy_y]
        )
        return losses

    def _print_output(self):
        g_loss = np.round(np.mean(self.history["G_loss"]), decimals=3)
        d_loss_p = np.round(
            np.mean(self.history["D_loss_positives"]), decimals=3
        )
        d_loss_n = np.round(
            np.mean(self.history["D_loss_negatives"]), decimals=3
        )
        d_loss_d = np.round(
            np.mean(self.history["D_loss_dummies"]), decimals=3
        )
        logger.info(
            f"Images shown {self.images_shown}:"
            + f" Generator Loss: {g_loss}"
            + f" - Discriminator Loss + : {d_loss_p}"
            + f" - Discriminator Loss - : {d_loss_n}"
            + f" - Discriminator Loss Dummies : {d_loss_d}"
        )
