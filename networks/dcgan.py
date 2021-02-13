import os

import numpy as np
from tensorflow.python.keras import Input, Model, layers
from tensorflow.python.keras.losses import binary_crossentropy

from config import config
from networks.base import GAN
from networks.utils import plot_metric
from networks.utils.layers import MinibatchSd
from utils import logger
from utils.image_operations import generate_images

DATA_FORMAT = config["data_format"]


class DCGAN(GAN):
    def __init__(
        self,
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
            img_height: height of the image. Should be a power of 2
            img_width: width of the image. Should be a power of 2
            channels: Number of image channels. Normally either 1 or 3.
            latent_size: Size of the latent vector that is used to generate the
                image
        """
        super(DCGAN, self).__init__()
        self.img_height = np.int(img_height)
        self.img_width = np.int(img_width)
        self.channels = channels
        self.img_shape = (
            (self.img_height, self.img_width, self.channels)
            if DATA_FORMAT == "channels_last"
            else (self.channels, self.img_height, self.img_width)
        )
        self.latent_size = latent_size
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
        n_filters = self.img_width // 2
        noise_input = Input((self.latent_size,))

        x = layers.Dense(
            (self.img_width // 2) * 4 * 4, name="Generator_Dense"
        )(noise_input)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape(((self.img_width // 2), 4, 4))(x)
        x = layers.Conv2D(
            n_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            data_format=DATA_FORMAT,
        )(x)

        cur_img_size = 4
        n_filters = self.img_width // 2
        while cur_img_size < self.img_shape[2]:
            x = layers.UpSampling2D(data_format=DATA_FORMAT)(x)
            x = layers.Conv2D(
                n_filters,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="same",
                kernel_initializer="he_normal",
                data_format=DATA_FORMAT,
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            cur_img_size *= 2
            n_filters //= 2

        generator_output = layers.Conv2D(
            self.channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            kernel_initializer="he_normal",
            data_format=DATA_FORMAT,
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
        n_filters = 16
        x = layers.Conv2D(
            n_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer="he_normal",
            data_format=DATA_FORMAT,
        )(image_input)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.3)(x)

        cur_img_size = self.img_shape[2] // 2
        while cur_img_size > 4:
            x = layers.Conv2D(
                n_filters,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="same",
                kernel_initializer="he_normal",
                data_format=DATA_FORMAT,
            )(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.3)(x)
            n_filters *= 2
            cur_img_size //= 2

        x = MinibatchSd()(x)
        x = layers.Flatten()(x)
        discriminator_output = layers.Dense(
            1, kernel_initializer="he_normal", activation="sigmoid"
        )(x)
        discriminative_model = Model(image_input, discriminator_output)
        return discriminative_model

    def train_on_batch(self, real_images: np.ndarray):
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
        self.metrics["D_accuracy"]["values"].append(discriminator_batch_acc)

        self.metrics["G_loss"]["values"].append(
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
            self.train_on_batch(batch)
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
