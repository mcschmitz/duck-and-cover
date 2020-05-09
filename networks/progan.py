"""
Definition of the ProGAN class.
"""


import copy
import functools
import os

import numpy as np
from keras import Model, Sequential
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import AveragePooling2D, Flatten, Input, LeakyReLU, Reshape, UpSampling2D

from networks import GAN
from networks.utils import PixelNorm, gradient_penalty_loss, save_gan, wasserstein_loss
from networks.utils.layers import MinibatchSd, RandomWeightedAverage, ScaledConv2D, ScaledDense, WeightedSum
from utils import generate_images, plot_metric

#  TODO Add release year information
#  TODO Add genre information
#  TODO Add artist
#  TODO add album name


class ProGAN(GAN):
    """
    Progressive growing GAN.

    Progressive Growing GAN that iteratively adds convolutional blocks to generator and discriminator. This results
    in improved quality, stability, variation and a faster learning time. Initialization itself is similar to the a
    normal Wasserstein GAN. Training, however, is slightly different: Initialization of the training phase should be
    done on a small scale image (e.g. 4x4). After an initial burn-in phase (train the GAN on the current resolution), a
    fade-in phase follows; meaning that on fades in the 8x8 upscale layer using the add add_fade_in_layers method. This
    phase is followed by another burn-in phase on the current image resolution. The procedure is repeated until the
    output image has the desired resolution.

    Args:
        gradient_penalty_weight: Weight for the gradient penalty loss.
        latent_size: Size of the latent vector that is used to generate the image.
    """

    def __init__(self, gradient_penalty_weight: int = 10, latent_size: int = 256):
        super(ProGAN, self).__init__()
        self.img_shape = ()
        self.channels = None
        self.latent_size = latent_size
        self.discriminator_loss = []
        self._gradient_penalty_weight = gradient_penalty_weight
        self.batch_size = None
        self.discriminator_model = []

    def build_models(
        self, optimizer, batch_size: list = None, channels: int = None, n_blocks: int = 1,
    ):
        """
        Builds the desired GAN that allows to generate covers.

        Builds the generator, the discriminator and the combined model for a WGAN using Wasserstein loss with gradient
        penalty to improve learning. Iteratively adds new generator and discriminator blocks to the GAN to improve
        the learning.

        Args:
            optimizer: Which optimizer to use for the combined model
            channels: number of channels of the output image
            batch_size: batch size of the GAN
            n_blocks: Number of blocks to add. each block doubles the size of the output image starting by 4*4. So
                n_blocks=1 will result in an image of size 8*8.
        """
        img_height = img_width = 2 ** (n_blocks + 1)
        self.channels = channels if channels is not None else self.channels
        self.img_shape = (img_height, img_width, self.channels)
        self.batch_size = batch_size if batch_size is not None else self.batch_size

        self.discriminator = self._build_discriminator(n_blocks, optimizer=optimizer)
        self.generator = self._build_generator(n_blocks)

        self._build_discriminator_model(optimizer, n_blocks)

        for block in range(n_blocks):
            for i in [0, 1]:
                for layer in self.discriminator_model[block][i].layers:
                    layer.trainable = False
                self.discriminator_model[block][i].trainable = False

        self.history["D_loss"] = []
        self.history["D_loss_positives"] = []
        self.history["D_loss_negatives"] = []
        self.history["D_loss_dummies"] = []
        self.history["G_loss"] = []

        self.combined_model = self._build_combined_model(optimizer)

    def _build_discriminator_model(self, optimizer, n_blocks):
        for block in range(n_blocks):
            discriminator_models = []
            for i in [0, 1]:
                disc_input_shp = self.discriminator[block][i].input_shape[1:]
                real_input = Input(disc_input_shp, name="Real_Img_Input")
                fake_input = Input(disc_input_shp, name="Fake_Img_Input")
                disc_fake = self.discriminator[block][i](fake_input)
                disc_real = self.discriminator[block][i](real_input)
                avg_samples = RandomWeightedAverage(self.batch_size[block])([real_input, fake_input])
                disc_avg_disc = self.discriminator[block][i](avg_samples)
                discriminator_model = Model(
                    inputs=[real_input, fake_input], outputs=[disc_real, disc_fake, disc_avg_disc],
                )
                partial_gp_loss = functools.partial(
                    gradient_penalty_loss,
                    averaged_samples=avg_samples,
                    gradient_penalty_weight=self._gradient_penalty_weight,
                )
                partial_gp_loss.__name__ = "gradient_penalty"
                optimizer = copy.copy(optimizer)
                discriminator_model.compile(
                    loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], optimizer=optimizer,
                )
                discriminator_models.append(discriminator_model)
            self.discriminator_model.append(discriminator_models)

    def _build_combined_model(self, optimizer):
        model_list = list()

        for idx, _ in enumerate(self.discriminator):
            g_models, d_models = self.generator[idx], self.discriminator[idx]
            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            optimizer = copy.copy(optimizer)
            model1.compile(loss=wasserstein_loss, optimizer=optimizer)

            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            optimizer = copy.copy(optimizer)
            model2.compile(loss=wasserstein_loss, optimizer=optimizer)

            model_list.append([model1, model2])
        return model_list

    def _build_generator(self, n_blocks):
        init = RandomNormal(0, 1)
        model_list = list()
        n_filters = self.latent_size

        latent_input = Input(shape=(self.latent_size,))
        x = ScaledDense(units=4 * 4 * self.latent_size, kernel_initializer=init, gain=np.sqrt(2) / 4)(latent_input)
        x = Reshape((4, 4, self.latent_size))(x)

        for _ in range(2):
            x = ScaledConv2D(filters=n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init,)(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = PixelNorm()(x)

        out_image = ScaledConv2D(
            filters=self.channels, kernel_size=(1, 1), padding="same", kernel_initializer=init, gain=1,
        )(x)

        model = Model(latent_input, out_image)
        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            models = self._add_generator_block(old_model, block=i)
            model_list.append(models)
        return model_list

    def _build_discriminator(self, n_blocks, optimizer, input_shape: tuple = (4, 4, 3)):
        init = RandomNormal(0, 1)
        model_list = list()
        n_filters = self._calc_filters(4)
        image_input = Input(input_shape)

        x = ScaledConv2D(filters=n_filters, kernel_size=(1, 1), padding="same", kernel_initializer=init,)(image_input)
        x = LeakyReLU(0.2)(x)

        x = MinibatchSd()(x)
        x = ScaledConv2D(filters=n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init,)(x)
        x = LeakyReLU(0.2)(x)

        x = ScaledConv2D(filters=n_filters, kernel_size=(4, 4), padding="same", kernel_initializer=init,)(x)
        x = LeakyReLU(0.2)(x)

        x = Flatten()(x)
        x = ScaledDense(units=1, gain=1)(x)

        model = Model(image_input, x)
        optimizer = copy.copy(optimizer)
        model.compile(loss=wasserstein_loss, optimizer=optimizer)
        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            optimizer = copy.copy(optimizer)
            models = self._add_discriminator_block(old_model, optimizer=optimizer)
            model_list.append(models)
        return model_list

    def train(self, data_loader, block, steps, batch_size, fade, **kwargs):
        """
        @TODO
        Args:
            data_loader:
            block:
            steps:
            batch_size:
            fade:
            **kwargs:

        Returns:

        """
        verbose = kwargs.get("verbose", False)
        model_dump_path = kwargs.get("write_model_to", None)
        minibatch_reps = kwargs.get("minibatch_reps", 1)
        n_critic = kwargs.get("n_critic", 1)
        path = kwargs.get("path", ".")

        initial_iter = len(self.history["G_loss"])
        alphas = np.linspace(0, 1, steps).tolist()
        f_idx = 1 if fade else 0
        gan = self.combined_model[block][f_idx]
        discriminator = self.discriminator_model[block][f_idx]
        generator = self.generator[block][f_idx]

        for step in range(initial_iter // batch_size, steps):

            if fade:
                alpha = alphas.pop(0)
                self._update_alpha(alpha, block)
            batch = data_loader.get_next_batch(batch_size)

            for _ in range(minibatch_reps):
                self.train_on_batch(gan, discriminator, generator, batch, n_critic=n_critic)
            self.images_shown += batch_size

            if step % (steps // 10) == 0 and verbose:
                self._print_output()

                final_resolution = self.img_shape[0]
                target_size = (final_resolution * 9, final_resolution * 9)
                img_path = os.path.join(path, "step{}.png".format(self.images_shown))
                generate_images(generator, img_path, n_imgs=9, target_size=target_size)

                img_path = os.path.join(path, "fixed_step{}.png".format(self.images_shown))
                generate_images(generator, img_path, n_imgs=9, target_size=target_size, seed=101)

                if fade:
                    for a in [0, 1]:
                        self._update_alpha(a, block)
                        img_path = os.path.join(path, "fixed_step{}_alpha{}.png".format(self.images_shown, a))
                        generate_images(generator, img_path, n_imgs=9, target_size=target_size, seed=101)

                metric = [
                    self.history["D_loss"],
                    self.history["D_loss_positives"],
                    self.history["D_loss_negatives"],
                    self.history["D_loss_dummies"],
                    self.history["G_loss"],
                ]
                file_names = ["d_loss.png", "d_lossP.png", "d_lossN.png", "d_lossD.png", "g_loss.png"]
                labels = [
                    "Discriminator Loss.png",
                    "Discriminator Loss (positives).png",
                    "Discriminator Loss (" "negatives).png",
                    "Discriminator Loss (Dummies).png",
                    "Generator Loss.png",
                ]

                for metric, file_name, label in zip(metric, file_names, labels):
                    plot_metric(path, steps=self.images_shown, metric=metric, y_label=label, file_name=file_name)

                if model_dump_path:
                    save_gan(self, model_dump_path)

    def train_on_batch(self, gan, discriminator, generator, real_images, n_critic: int = 5):
        """
        Runs a single gradient update on a batch of data.

        @ TODO
        Args:
            real_images: numpy array of real input images used for training
            n_critic: number of discriminator updates for each iteration
        """
        batch_size = real_images.shape[0] // n_critic
        real_y = np.ones((batch_size, 1)) * -1

        for i in range(n_critic):
            discriminator_minibatch = real_images[i * batch_size : (i + 1) * batch_size]
            losses = self.train_discriminator(discriminator, generator, discriminator_minibatch)

        self.history["D_loss"].append(losses[0])
        self.history["D_loss_positives"].append(losses[1])
        self.history["D_loss_negatives"].append(losses[2])
        self.history["D_loss_dummies"].append(losses[3])

        noise = np.random.normal(size=(batch_size, self.latent_size))
        self.history["G_loss"].append(gan.train_on_batch(noise, real_y))

    def _print_output(self):
        print(
            "Images shown {0}: Generator Loss: {1:3,.3f} - Discriminator Loss: {2:3,.3f} - "
            "Discriminator Loss + : {2:3,.3f} - Discriminator Loss - : {3:3,.3f} -"
            " Discriminator Loss Dummies : {4:3,.3f}".format(
                self.images_shown,
                np.mean(self.history["G_loss"]),
                np.mean(self.history["D_loss"]),
                np.mean(self.history["D_loss_positives"]),
                np.mean(self.history["D_loss_negatives"]),
                np.mean(self.history["D_loss_dummies"]),
            )
        )

    def train_discriminator(self, discriminator, generator, x):
        """
        Runs a single gradient update on a batch of data.

        @ TODO
        Args:
            x: numpy array of real input images used for training

        Returns:
            the losses for this training iteration
        """
        batch_size = len(x)
        fake_y = np.ones((batch_size, 1))
        real_y = np.ones((batch_size, 1)) * -1
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
        noise = np.random.normal(size=(batch_size, self.latent_size))
        fake_x = generator.predict(noise)
        losses = discriminator.train_on_batch([x, fake_x], [real_y, fake_y, dummy_y])
        return losses

    def _update_alpha(self, alpha, block):
        models = [
            self.generator[block][1],
            self.discriminator_model[block][1],
            self.combined_model[block][1],
        ]
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    K.set_value(layer.alpha, alpha)

    def _calc_filters(self, x: int):
        return int(min((4 * 4 * self.latent_size / x) * 2, self.latent_size))

    def _add_discriminator_block(self, old_model: Model, optimizer, n_input_layers: int = 3) -> list:
        n_filters = self.img_shape[0]

        init = RandomNormal(0, 1)
        in_shape = list(old_model.input.shape)
        input_shape = (
            in_shape[-2].value * 2,
            in_shape[-2].value * 2,
            in_shape[-1].value,
        )
        in_image = Input(shape=input_shape)

        d = ScaledConv2D(filters=n_filters, kernel_size=(1, 1), padding="same", kernel_initializer=init,)(in_image)
        d = LeakyReLU(alpha=0.2)(d)

        d = ScaledConv2D(filters=n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init,)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = ScaledConv2D(filters=n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init,)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D(2, 2)(d)
        block_new = d

        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)

        model1 = Model(in_image, d)
        model1.compile(loss=wasserstein_loss, optimizer=optimizer)

        downsample = AveragePooling2D(2, 2)(in_image)
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        d = WeightedSum()([block_old, block_new])

        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        model2 = Model(in_image, d)
        model2.compile(loss=wasserstein_loss, optimizer=optimizer)
        return [model1, model2]

    def _add_generator_block(self, old_model: Model, block: int) -> list:
        cur_resolution = 2 ** (2 + block)
        n_filters = self._calc_filters(cur_resolution)

        init = RandomNormal(0, 1)

        block_end = old_model.layers[-2].output
        upsampling = UpSampling2D()(block_end)
        g = ScaledConv2D(filters=n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init,)(upsampling)
        g = LeakyReLU(alpha=0.2)(g)
        g = PixelNorm()(g)

        g = ScaledConv2D(filters=n_filters, kernel_size=(3, 3), padding="same", kernel_initializer=init,)(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = PixelNorm()(g)

        out_image = ScaledConv2D(filters=self.channels, kernel_size=(1, 1), padding="same", kernel_initializer=init,)(g)
        model1 = Model(old_model.input, out_image)

        out_old = old_model.layers[-1]
        out_image2 = out_old(upsampling)
        merged = WeightedSum()([out_image2, out_image])

        model2 = Model(old_model.input, merged)
        return [model1, model2]
