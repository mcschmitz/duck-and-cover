"""
This script defines and trains the ProGAN as defined in
https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-
keras-for-synthesizing-faces/, whereby some functions and classes are modified
or enhanced.
"""

import numpy as np
from keras.initializers import RandomNormal
from keras.layers import AveragePooling2D, Flatten, Input, LeakyReLU
from keras.models import Model, Sequential
from tqdm import tqdm

from networks.utils.layers import MinibatchSd, WeightedSum, ScaledDense, ScaledConv2D
from networks.utils import wasserstein_loss
from utils import load_data
from keras_gradient_accumulation import GradientAccumulation
from keras.optimizers import Adam
from train.ml_mastery_progan import (
    define_generator,
    generate_real_samples,
    generate_fake_samples,
    generate_latent_points,
    update_fadein,
    summarize_performance,
)

BATCH_SIZE = 128

ACCUMULATIVE_UPDATES = [1, 1, 2, 4, 8, 16, 32]
optimizers = [
    GradientAccumulation(Adam(0.001, beta_1=0.0, beta_2=0.99), accumulation_steps=i) for i in ACCUMULATIVE_UPDATES
]
dist_optimizers = [
    GradientAccumulation(Adam(0.001, beta_1=0.0, beta_2=0.99), accumulation_steps=i) for i in ACCUMULATIVE_UPDATES
]


def define_discriminator(n_double: int, input_shape: tuple = (4, 4, 3)) -> list:
    """
    Defines a list of discriminator models.

    Args:
        n_double: Number of doublings of the image resolution
        input_shape: Shape of the input image

    Returns:
        list of discriminator models
    """
    init = RandomNormal(0, 1)
    model_list = list()

    in_image = Input(shape=input_shape)

    d = ScaledConv2D(filters=128, kernel_size=(1, 1), padding="same", kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = MinibatchSd()(d)
    d = ScaledConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ScaledConv2D(filters=128, kernel_size=(4, 4), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Flatten()(d)
    out_class = ScaledDense(units=1, gain=1)(d)

    model = Model(in_image, out_class)
    model.compile(loss=wasserstein_loss, optimizer=dist_optimizers[0])

    model_list.append([model, model])
    for i in range(1, n_double):
        old_model = model_list[i - 1][0]
        models = add_discriminator_block(old_model, block=i)
        model_list.append(models)
    return model_list


def add_discriminator_block(old_model: Model, n_input_layers: int = 3, block: int = 0) -> list:
    """
    Adds a new block to the discriminator model.

    Args:
        old_model: Already well trained discriminator model
        n_input_layers: Number of input layers in the discriminator model

    Returns:
        List of old and new model
    """
    init = RandomNormal(0, 1)

    in_shape = list(old_model.input.shape)
    input_shape = (in_shape[-2].value * 2, in_shape[-2].value * 2, in_shape[-1].value)
    in_image = Input(shape=input_shape)

    d = ScaledConv2D(filters=128, kernel_size=(1, 1), padding="same", kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = ScaledConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = ScaledConv2D(filters=128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D(2, 2)(d)
    block_new = d

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model1 = Model(in_image, d)
    model1.compile(loss=wasserstein_loss, optimizer=dist_optimizers[block])

    downsample = AveragePooling2D(2, 2)(in_image)
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    d = WeightedSum()([block_old, block_new])

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model2 = Model(in_image, d)
    model2.compile(loss=wasserstein_loss, optimizer=dist_optimizers[block])
    return [model1, model2]


def define_combined(discriminators: list, generators: list) -> list:
    """
    Build the combined models out of the given generator and discriminator
    models.

    Args:
        discriminators: list of discriminator models
        generators: list of generator models

    Returns:
        list of combined models
    """
    model_list = list()

    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=optimizers[i])

        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=optimizers[i])

        model_list.append([model1, model2])
    return model_list


def train_epochs(
    g_model: Model,
    d_model: Model,
    gan_model: Model,
    dataset: np.array,
    n_epochs: int,
    block: int,
    fadein: bool = False,
):
    """
    Trains the model.

    Args:
        g_model: Generator model
        d_model: Discriminator model
        gan_model: Combined model
        dataset: Trainset
        n_epochs: Number of epochs
        block: Block which is currently trained
        fadein: Fade in new layer?
    """
    batch_size = BATCH_SIZE // ACCUMULATIVE_UPDATES[block]
    n_batches = dataset.shape[0] // batch_size
    n_steps = n_batches * n_epochs
    half_batch = batch_size // 2
    for step in tqdm(range(n_steps)):
        if fadein:
            update_fadein([g_model, d_model, gan_model], step, n_steps)
        x_real, y_real = generate_real_samples(dataset, half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        x_disc = np.concatenate([x_real, x_fake])
        y_disc = np.concatenate([y_real, y_fake])
        d_model.train_on_batch(x_disc, y_disc)
        z_input = generate_latent_points(latent_dim, batch_size)
        y_real2 = np.ones((batch_size, 1))
        gan_model.train_on_batch(z_input, y_real2)


def train(g_models, d_models, gan_models, dataset, latent_dim, epochs):
    """
    Trains the combined GAN.

    Args:
        g_models: List of generator models
        d_models: List of discriminator models
        gan_models: List of combined GAN models
        dataset: Dataset
        latent_dim: Dimension of the latent vector
        epochs: Number of epochs
    """
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]

    gen_shape = g_normal.output_shape
    scaled_data, _ = load_data(dataset, gen_shape[1])

    print("\n\nBurn in model_{0}".format(gen_shape))
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, epochs[0], 1)
    summarize_performance("tuned", g_normal, latent_dim)

    for i in range(1, len(g_models)):
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]

        gen_shape = g_normal.output_shape
        scaled_data, _ = load_data(dataset, gen_shape[1])

        print("\n\nFade in model_{0}".format(gen_shape))
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, epochs[i], i, True)
        summarize_performance("faded", g_fadein, latent_dim)

        print("\n\nBurn in model_{0}".format(gen_shape))
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, epochs[i], i)
        summarize_performance("tuned", g_normal, latent_dim)


n_blocks = 6
latent_dim = 128
d_models = define_discriminator(n_blocks)
g_models = define_generator(latent_dim, n_blocks)
gan_models = define_combined(d_models, g_models)
data_path = "data/celeba"
n_epochs = [10, 10, 10, 10, 10, 10]
train(g_models, d_models, gan_models, data_path, latent_dim, n_epochs)
