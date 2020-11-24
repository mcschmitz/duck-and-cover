import copy
import itertools
import os

import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    AveragePooling2D,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    UpSampling2D,
)

from networks.utils import save_gan, wasserstein_loss
from networks.utils.layers import (
    GradientPenalty,
    MinibatchSd,
    PixelNorm,
    RandomWeightedAverage,
    ScaledConv2D,
    ScaledDense,
    WeightedSum,
)
from networks.wgan import WGAN
from utils import generate_images, plot_metric

#  TODO Add release year information
#  TODO Add genre information
#  TODO Add artist
#  TODO add album name


class ProGAN(WGAN):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        gradient_penalty_weight: float = 10,
        latent_size: int = 128,
        channels: int = 3,
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
        super(ProGAN, self).__init__(
            img_height=img_height,
            img_width=img_width,
            channels=channels,
            latent_size=latent_size,
            gradient_penalty_weight=gradient_penalty_weight,
        )
        self.img_shape = ()
        self.discriminator_model = []

    def build_models(
        self,
        optimizer,
        n_blocks: int = 1,
    ):
        """
        Builds the desired GAN that allows to generate covers.

        Builds the generator, the discriminator and the combined model for a
        WGAN using Wasserstein loss with gradient penalty to improve learning.
        Iteratively adds new generator and discriminator blocks to the GAN to
        improve
        the learning.

        Args:
            optimizer: Which optimizer to use for the combined model
            n_blocks: Number of blocks to add. each block doubles the size of
                the output image starting by 4*4. So n_blocks=1 will result in
                an image of size 8*8.
        """
        self.generator = self._build_generator(n_blocks)
        fade = {0, 1}
        for block, f in itertools.product(range(n_blocks), fade):
            self.generator[block][f].trainable = False
        self.discriminator = self._build_discriminator(
            n_blocks, optimizer=optimizer
        )
        self._build_discriminator_model(optimizer, n_blocks)
        self.combined_model = self._build_combined_model(optimizer)

    def _build_discriminator_model(self, optimizer, n_blocks):
        fade = {0, 1}
        for block in range(n_blocks):
            discriminator_models = []
            for f in fade:
                discriminator = self.discriminator[block][f]
                generator = self.generator[block][f]
                disc_input_shp = list(discriminator.input.shape[1:])
                disc_input_image = Input(disc_input_shp, name="Img_Input")
                disc_input_noise = Input(
                    (self.latent_size,), name="Noise_Input_for_Discriminator"
                )
                gen_image_noise = generator(disc_input_noise)
                disc_image_noise = discriminator(gen_image_noise)
                disc_image_real = discriminator(disc_input_image)
                avg_samples = RandomWeightedAverage()(
                    [disc_input_image, gen_image_noise]
                )
                disc_avg = discriminator(avg_samples)
                gp_layer = GradientPenalty(
                    weight=self._gradient_penalty_weight
                )
                gp = gp_layer([disc_avg, avg_samples])

                discriminator_model = Model(
                    inputs=[disc_input_image, disc_input_noise],
                    outputs=[disc_image_real, disc_image_noise, gp],
                )
                optimizer = copy.copy(optimizer)
                discriminator_model.compile(
                    loss=[wasserstein_loss, wasserstein_loss, "mse"],
                    optimizer=optimizer,
                )
                discriminator_models.append(discriminator_model)
            self.discriminator_model.append(discriminator_models)

    def _build_combined_model(self, optimizer):
        model_list = []

        for idx, _ in enumerate(self.discriminator):
            g_models, d_models = self.generator[idx], self.discriminator[idx]
            d_models[0].trainable = False
            g_models[0].trainable = True
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            optimizer = copy.copy(optimizer)
            model1.compile(loss=wasserstein_loss, optimizer=optimizer)

            d_models[1].trainable = False
            g_models[1].trainable = True
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
        x = ScaledDense(
            units=4 * 4 * self.latent_size,
            kernel_initializer=init,
            gain=np.sqrt(2) / 4,
        )(latent_input)
        x = Reshape((4, 4, self.latent_size))(x)

        for _ in range(2):
            x = ScaledConv2D(
                filters=n_filters,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=init,
            )(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = PixelNorm()(x)

        out_image = ScaledConv2D(
            filters=self.channels,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
            gain=1,
        )(x)

        model = Model(latent_input, out_image)
        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            models = self._add_generator_block(old_model, block=i)
            model_list.append(models)
        return model_list

    def _build_discriminator(
        self, n_blocks, optimizer, input_shape: tuple = (4, 4, 3)
    ):
        init = RandomNormal(0, 1)
        model_list = list()
        n_filters = self._calc_filters(4)
        image_input = Input(input_shape)

        x = ScaledConv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
        )(image_input)
        x = LeakyReLU(0.2)(x)

        x = MinibatchSd()(x)
        x = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
        )(x)
        x = LeakyReLU(0.2)(x)

        x = ScaledConv2D(
            filters=n_filters,
            kernel_size=(4, 4),
            padding="same",
            kernel_initializer=init,
        )(x)
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
            models = self._add_discriminator_block(
                old_model, optimizer=optimizer
            )
            model_list.append(models)
        return model_list

    def train(
        self, data_loader, block, global_steps, batch_size, fade, **kwargs
    ):
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
        path = kwargs.get("path", ".")
        steps = global_steps // batch_size
        verbose = kwargs.get("verbose", True)
        model_dump_path = kwargs.get("write_model_to", None)
        minibatch_reps = kwargs.get("minibatch_reps", 1)
        n_critic = kwargs.get("n_critic", 1)

        alphas = np.linspace(0, 1, steps).tolist()
        f_idx = 1 if fade else 0
        gan = self.combined_model[block][f_idx]
        discriminator = self.discriminator_model[block][f_idx]

        for step in range(self.images_shown // batch_size, steps):
            batch = data_loader.get_next_batch(batch_size)
            if fade:
                alpha = alphas.pop(0)
                self._update_alpha(alpha, block)

            for _ in range(minibatch_reps):
                self.train_on_batch(
                    gan, discriminator, batch, n_critic=n_critic
                )
            self.images_shown += batch_size

            if step % 250 == 0 and verbose:
                self._print_output()
                self._generate_images(path, block, f_idx)

                for _k, v in self.metrics.items():
                    plot_metric(
                        path,
                        steps=self.images_shown,
                        metric=v.get("values"),
                        y_label=v.get("label"),
                        file_name=v.get("file_name"),
                    )
                if model_dump_path:
                    save_gan(self, model_dump_path)

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

    def _add_discriminator_block(
        self, old_model: Model, optimizer, n_input_layers: int = 3
    ) -> list:
        n_filters = self.latent_size

        init = RandomNormal(0, 1)
        in_shape = list(old_model.input.shape)
        input_shape = (
            in_shape[-2] * 2,
            in_shape[-2] * 2,
            in_shape[-1],
        )
        in_image = Input(shape=input_shape)

        d = ScaledConv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
        )(in_image)
        d = LeakyReLU(alpha=0.2)(d)

        d = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
        )(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
        )(d)
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
        g = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
        )(upsampling)
        g = LeakyReLU(alpha=0.2)(g)
        g = PixelNorm()(g)

        g = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
        )(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = PixelNorm()(g)

        out_image = ScaledConv2D(
            filters=self.channels,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
        )(g)
        model1 = Model(old_model.input, out_image)

        out_old = old_model.layers[-1]
        out_image2 = out_old(upsampling)
        merged = WeightedSum()([out_image2, out_image])

        model2 = Model(old_model.input, merged)
        return [model1, model2]

    def _generate_images(self, path, block, fade):
        for s in range(25):
            img_path = os.path.join(
                path, f"{s}_fixed_step_gif{self.images_shown}.png"
            )
            generate_images(
                self.generator[block][fade],
                img_path,
                target_size=(256, 256),
                seed=s,
                n_imgs=1,
            )

        if fade:
            for a in [0, 1]:
                self._update_alpha(a, block)
                img_path = os.path.join(
                    path,
                    "fixed_step{}_alpha{}.png".format(self.images_shown, a),
                )
                generate_images(
                    self.generator[block][fade],
                    img_path,
                    target_size=(256, 256),
                    seed=101,
                )
