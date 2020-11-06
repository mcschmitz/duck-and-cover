import logging
import os

import numpy as np
from tensorflow.python.keras import Input, Model, layers
from tensorflow.python.keras.losses import binary_crossentropy

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL
from networks.base import GAN
from networks.utils.layers import MinibatchSd
from utils import generate_images, plot_metric

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger(__file__)


class DCGAN(GAN):
    def __init__(
        self,
        batch_size: int,
        img_height: int,
        img_width: int,
        channels: int = 3,
        latent_size: int = 128,
    ):
        """
        Simple Deep Convolutional GAN.

        Simple DCGAN that builds a discriminator a generator and the adversarial
        model to train the GAN based on the binary crossentropy loss for the
        generator and the discriminator.

        Args:
            batch_size: size of the training batches
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the
                image
        """
        super(DCGAN, self).__init__()
        self.batch_size = batch_size
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (self.img_height, self.img_width, self.channels)
        self.latent_size = latent_size
        self.discriminator_accuracy = []
        self.critic_below50 = 0

    def build_models(self, combined_optimizer, discriminator_optimizer=None):
        """
        Builds the desired GAN that allows to generate covers.

        Creates every model needed for GAN. Creates a discriminator, a generator
        and the combined model. The discriminator as well as the generator are
        trained using the provided optimizer.

        Args:
            combined_optimizer: Which optimizer to use for the combined model
            discriminator_optimizer: Which optimizer to use for the
                discriminator model
        """
        discriminator_optimizer = (
            combined_optimizer
            if discriminator_optimizer is None
            else discriminator_optimizer
        )
        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        for layer in self.generator.layers:
            layer.trainable = False
        self.generator.trainable = False
        self._build_discriminator_model(discriminator_optimizer)
        self.history["D_accuracy"] = []

        self.fuse_disc_and_gen(combined_optimizer)

    def _build_discriminator_model(self, optimizer):
        """
        Build the discriminator model.

        The discriminator model is only compiled in this step since the model is
        rather simple in a plain GAN

        Args:
            optimizer: Which optimizer to use
        """
        self.discriminator.compile(
            loss=[binary_crossentropy],
            optimizer=optimizer,
            metrics=["accuracy"],
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
        self.combined_model.compile(optimizer, loss=[binary_crossentropy])

    def _build_generator(self):
        """
        Builds the DCGAN generator.

        Builds the very simple generator that takes a latent input vector and
        applies the following block until the desired image size is reached:
        3x3 convolutional layer with ReLu activation -> Batch Normalization ->
        Upsamling layer. The last Convolutional layer wit tanH activation
        results in 3 RGB channels and serves as final output

        Returns:
            The DCGAN generator
        """
        noise_input = Input((self.latent_size,))
        x = layers.Dense(4 * 4 * 512, name="Generator_Dense")(noise_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((4, 4, 512))(x)
        x = layers.Conv2D(
            512,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
        )(x)

        cur_img_size = 4
        n_channels = 256
        while cur_img_size < self.img_shape[0]:
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(
                n_channels,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            cur_img_size *= 2
            n_channels //= 2

        generator_output = layers.Conv2D(
            self.channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        generator_output = layers.Activation("tanh")(generator_output)
        generator_model = Model(noise_input, generator_output)
        return generator_model

    def _build_discriminator(self):
        """
        Builds the DCGAN discriminator.

        Builds the very simple discriminator that takes an image input and
        applies a 3x3 convolutional layer with ReLu activation and a 2x2 stride
        until the desired embedding  size is reached. The flattend embedding is
        ran through a Dense layer with sigmoid output to label the image.

        Returns:
            The DCGAN discriminator
        """
        image_input = Input(self.img_shape)
        x = layers.Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer="he_normal",
        )(image_input)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        cur_img_size = self.img_shape[0] // 2
        n_channels = 64
        while cur_img_size > 4:
            x = layers.Conv2D(
                64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer="he_normal",
            )(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.3)(x)
            n_channels *= 2
            cur_img_size //= 2

        x = MinibatchSd()(x)
        x = layers.Flatten()(x)
        discriminator_output = layers.Dense(
            1, kernel_initializer="he_normal", activation="sigmoid"
        )(x)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def train_on_batch(self, real_images: np.array):
        """
        Runs a single gradient update on a batch of data.

        Args:
            real_images: numpy array of real input images used for training
        """
        fake = np.ones(len(real_images))
        real = np.zeros(len(real_images))

        noise = np.random.normal(size=(len(real_images), self.latent_size))
        generated_images = self.generator.predict(noise)

        discriminator_x = np.concatenate((generated_images, real_images))
        discriminator_y = np.concatenate((fake, real))

        self.train_discriminator(generated_images, real_images)
        discriminator_batch_acc = self.discriminator.evaluate(
            discriminator_x, discriminator_y, verbose=0
        )[1]
        self.history["D_accuracy"].append(discriminator_batch_acc)

        self.history["G_loss"].append(
            self.combined_model.train_on_batch(noise, real)
        )

    def train_discriminator(
        self, generated_images: np.array, real_images: np.array
    ):
        """
        Runs a single gradient update for the discriminator.

        Args:
            generated_images: array of generated images by the generator
            real_images: array of real images
        """
        fake = np.ones(len(generated_images))
        real = np.zeros(len(real_images))
        self.discriminator.train_on_batch(generated_images, fake)
        self.discriminator.train_on_batch(real_images, real)

    def train(self, data_loader, global_steps, batch_size, **kwargs):
        path = kwargs.get("path", ".")
        steps = global_steps // batch_size
        for step in range(self.images_shown // batch_size, steps):
            batch = data_loader.get_next_batch(batch_size)
            self.train_on_batch(batch)
            self.images_shown += batch_size

            if step % 250 == 0:
                self._print_output()
                self._generate_images(path)

                metric = [self.history["D_accuracy"], self.history["G_loss"]]
                file_names = ["d_acc.png", "g_loss.png"]
                labels = ["Discriminator Accuracy.png", "Generator Loss.png"]

                for metric, file_name, label in zip(
                    metric, file_names, labels
                ):
                    plot_metric(
                        path,
                        steps=self.images_shown,
                        metric=metric,
                        y_label=label,
                        file_name=file_name,
                    )

    def _generate_images(self, path):
        img_path = os.path.join(path, f"step{self.images_shown}.png")
        generate_images(self.generator, img_path, target_size=(64, 64))

        img_path = os.path.join(path, f"fixed_step{self.images_shown}.png")
        generate_images(
            self.generator, img_path, target_size=(64, 64), seed=101
        )

        img_path = os.path.join(path, f"fixed_step_gif{self.images_shown}.png")
        generate_images(
            self.generator,
            img_path,
            target_size=(256, 256),
            seed=101,
            n_imgs=1,
        )

    def _print_output(self):
        g_loss = np.round(np.mean(self.history["G_loss"]), decimals=3)
        d_acc = np.round(np.mean(self.history["D_accuracy"]), decimals=3)
        logger.info(
            f"Images shown {self.images_shown}:"
            + f" Generator Loss: {g_loss} -"
            + f" Discriminator Acc.: {d_acc}"
        )
