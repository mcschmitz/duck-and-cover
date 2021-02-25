import copy
import os

import numpy as np
from defaultlist import defaultlist
from tensorflow.keras import Model
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
from tensorflow.keras.losses import mse

from config import config
from networks.gradient_accumulator_model import (
    GradientAccumulatorModel,
    GradientAccumulatorSequential,
)
from networks.utils import plot_metric, save_gan, wasserstein_loss
from networks.utils.layers import (
    GetGradients,
    MinibatchSd,
    PixelNorm,
    RandomWeightedAverage,
    ScaledConv2D,
    ScaledDense,
    WeightedSum,
)
from networks.wgan import WGAN
from utils import logger
from utils.image_operations import generate_images

DATA_FORMAT = config["data_format"]

#  TODO Add release year information
#  TODO Add genre information
#  TODO Add artist name
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
        self.block_images_shown = {
            "burn_in": defaultlist(int),
            "fade_in": defaultlist(int),
        }

    def build_models(
        self,
        optimizer,
        n_blocks: int = 1,
        gradient_accumulation_steps: int = 1,
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
            gradient_accumulation_steps: gradient accumulation steps that
                should be done when trainig.
        """
        self.generator = self._build_generator(n_blocks)
        self.discriminator = self._build_discriminator(
            n_blocks, optimizer=optimizer
        )
        for model in self.generator:
            model.trainable = False
        self.discriminator_model = self._build_discriminator_model(
            optimizer, n_blocks
        )
        self.combined_model = self._build_combined_model(
            optimizer,
            grad_acc_steps=gradient_accumulation_steps,
        )

    def _build_discriminator_model(self, optimizer, n_blocks, grad_acc_steps):
        fade = {0, 1}
        discriminator_models = []
        for f in fade:
            discriminator = self.discriminator[f]
            generator = self.generator[f]
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
            gradients = GetGradients(model=discriminator)(avg_samples)
            gp = self._gradient_penalty(gradients)
            discriminator_model = GradientAccumulatorModel(
                inputs=[disc_input_image, disc_input_noise],
                outputs=[disc_image_real, disc_image_noise, gp],
                gradient_accumulation_steps=grad_acc_steps,
            )
            optimizer = copy.copy(optimizer)
            discriminator_model.compile(
                loss=[wasserstein_loss, wasserstein_loss, mse],
                optimizer=optimizer,
            )
            discriminator_models.append(discriminator_model)
        return discriminator_models

    def _build_combined_model(self, optimizer, grad_acc_steps):
        self.discriminator[0].trainable = False
        self.generator[0].trainable = True
        model1 = GradientAccumulatorSequential(
            gradient_accumulation_steps=grad_acc_steps
        )
        model1.add(self.generator[0])
        model1.add(self.discriminator[0])
        optimizer = copy.copy(optimizer)
        model1.compile(loss=wasserstein_loss, optimizer=optimizer)

        self.discriminator[1].trainable = False
        self.generator[1].trainable = True
        model2 = GradientAccumulatorSequential(
            gradient_accumulation_steps=grad_acc_steps
        )
        model2.add(self.generator[1])
        model2.add(self.discriminator[1])
        optimizer = copy.copy(optimizer)
        model2.compile(loss=wasserstein_loss, optimizer=optimizer)

        return [model1, model2]

    def _build_generator(self, n_blocks) -> list:
        init = RandomNormal(0, 1)
        model_list = []
        n_filters = self.img_width // 2

        latent_input = Input(shape=self.latent_size)
        x = ScaledDense(
            units=4 * 4 * (self.img_width // 2),
            kernel_initializer=init,
            gain=np.sqrt(2) / 4,
        )(latent_input)
        x = Reshape((self.img_width // 2, 4, 4))(x)

        for _ in range(2):  # noqa: WPS122
            x = ScaledConv2D(
                filters=n_filters,
                kernel_size=(3, 3),
                padding="same",
                kernel_initializer=init,
                data_format=DATA_FORMAT,
            )(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = PixelNorm()(x)

        out_image = ScaledConv2D(
            filters=self.channels,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
            gain=1,
            data_format=DATA_FORMAT,
        )(x)

        model = Model(latent_input, out_image)
        model_list.append([model, model])

        for i in range(1, n_blocks):
            old_model = model_list[i - 1][0]
            models = self._add_generator_block(old_model, block=i)
            model_list.append(models)
        return model_list[-1]

    def _build_discriminator(
        self, n_blocks, optimizer, input_shape: tuple = (3, 4, 4)
    ) -> list:
        init = RandomNormal(0, 1)
        model_list = []
        n_filters = self._calc_filters(4)
        image_input = Input(input_shape)

        x = ScaledConv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
        )(image_input)
        x = LeakyReLU(0.2)(x)

        x = MinibatchSd()(x)
        x = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
        )(x)
        x = LeakyReLU(0.2)(x)

        x = ScaledConv2D(
            filters=n_filters,
            kernel_size=(4, 4),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
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
        return model_list[-1]

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
        f_idx = 1 if phase == "fade_in" else 0
        for step in range(phase_images_shown // batch_size, steps):
            batch = data_loader.__getitem__(step)
            if phase == "fade_in":
                alpha = alphas.pop(0)
                self._update_alpha(alpha, block)

            self.train_on_batch(batch, n_critic=n_critic, fade_idx=f_idx)
            self.images_shown += batch_size
            if f_idx:
                self.block_images_shown["fade_in"][block] += batch_size
            else:
                self.block_images_shown["burn_in"][block] += batch_size
            if step % (steps // 32) == 0 and verbose:
                self._print_output()
                self._generate_images(path, f_idx)

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
        save_gan(self, model_dump_path)
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
        return int(
            min((4 * 4 * (self.img_width // 2) / x) * 2, self.img_width // 2)
        )

    def _add_discriminator_block(
        self,
        old_model: GradientAccumulatorModel,
        optimizer,
        n_input_layers: int = 3,
    ) -> list:
        n_filters = self.img_width // 2

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
            data_format=DATA_FORMAT,
        )(in_image)
        d = LeakyReLU(alpha=0.2)(d)

        d = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
        )(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
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

    def _add_generator_block(
        self, old_model: GradientAccumulatorModel, block: int
    ) -> list:
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
            data_format=DATA_FORMAT,
        )(upsampling)
        g = LeakyReLU(alpha=0.2)(g)
        g = PixelNorm()(g)

        g = ScaledConv2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
        )(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = PixelNorm()(g)

        out_image = ScaledConv2D(
            filters=self.channels,
            kernel_size=(1, 1),
            padding="same",
            kernel_initializer=init,
            data_format=DATA_FORMAT,
        )(g)
        model1 = Model(old_model.input, out_image)

        out_old = old_model.layers[-1]
        out_image2 = out_old(upsampling)
        merged = WeightedSum()([out_image2, out_image])

        model2 = Model(old_model.input, merged)
        return [model1, model2]

    def _generate_images(self, path, fade):
        for s in range(25):
            img_path = os.path.join(
                path, f"{s}_fixed_step_gif{self.images_shown}.png"
            )
            generate_images(
                self.generator[fade],
                img_path,
                target_size=(256, 256),
                seed=s,
                n_imgs=1,
            )

        if fade:
            for a in [0, 1]:
                self._update_alpha(a)
                img_path = os.path.join(
                    path,
                    "fixed_step{}_alpha{}.png".format(self.images_shown, a),
                )
                generate_images(
                    self.generator[fade],
                    img_path,
                    target_size=(256, 256),
                    seed=101,
                )
