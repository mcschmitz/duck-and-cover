"""
This script defines and trains the ProGAN as defined in
https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-
keras-for-synthesizing-faces/, whereby some functions and classes are modified
or enhanced.
"""

from tqdm import tqdm
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from networks.utils import wasserstein_loss, gradient_penalty_loss
from networks.utils.layers import RandomWeightedAverage
from utils import load_data
import functools
from train.ml_mastery_progan import (
    update_fadein,
    generate_real_samples,
    generate_fake_samples,
    generate_latent_points,
    summarize_performance,
    define_discriminator,
    define_generator,
    define_combined,
)


def define_discriminator_models(discriminators: list, n_blocks, batch_size: list) -> list:
    disc_models = []
    for block in range(n_blocks):
        block_disc_models = []
        for i in [0, 1]:
            disc_input_shp = discriminators[block][i].input_shape[1:]
            real_input = Input(disc_input_shp, name="real_input")
            fake_input = Input(disc_input_shp, name="fake_input")
            disc_image_gen = discriminators[block][i](fake_input)
            disc_image_image = discriminators[block][i](real_input)
            avg_samples = RandomWeightedAverage(batch_size[block])([real_input, fake_input])
            disc_avg_disc = discriminators[block][i](avg_samples)
            disc_model = Model(
                inputs=[real_input, fake_input], outputs=[disc_image_image, disc_image_gen, disc_avg_disc],
            )
            partial_gp_loss = functools.partial(
                gradient_penalty_loss, averaged_samples=avg_samples, gradient_penalty_weight=10,
            )
            partial_gp_loss.__name__ = "gradient_penalty"
            disc_model.compile(
                loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8),
            )
            block_disc_models.append(disc_model)
        disc_models.append(block_disc_models)
    return disc_models


def train_epochs(
    g_model: Model,
    d_model: Model,
    gan_model: Model,
    dataset: np.array,
    n_epochs: int,
    batch_size: int,
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
        batch_size: Batch size
        fadein: Fade in new layer?
    """
    n_batches = dataset.shape[0] // batch_size
    n_steps = n_batches * n_epochs
    for i in tqdm(range(n_steps)):
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        x_real, y_real = generate_real_samples(dataset, batch_size)
        x_fake, y_fake = generate_fake_samples(g_model, LATENT_DIM, batch_size)
        y_dummy = np.zeros((batch_size, 1), dtype=np.float32)
        d_model.train_on_batch([x_real, x_fake], [y_real, y_fake, y_dummy])
        z_input = generate_latent_points(LATENT_DIM, batch_size)
        y_real2 = np.ones((batch_size, 1))
        gan_model.train_on_batch(z_input, y_real2)


def train(g_models, d_models, gan_models, dataset, latent_dim, epochs, n_batch):
    """
    Trains the combined GAN.

    Args:
        g_models: List of generator models
        d_models: List of discriminator models
        gan_models: List of combined GAN models
        dataset: Dataset
        latent_dim: Dimension of the latent vector
        epochs: Number of epochs
        n_batch: Batch size
    """
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]

    gen_shape = g_normal.output_shape
    scaled_data, _ = load_data(dataset, gen_shape[1])

    print("\n\nBurn in model_{0}".format(gen_shape))
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, epochs[0], n_batch[0])
    summarize_performance("tuned", g_normal, latent_dim)

    for i in range(1, len(g_models)):
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]

        gen_shape = g_normal.output_shape
        scaled_data, _ = load_data(dataset, gen_shape[1])

        print("\n\nFade in model_{0}".format(gen_shape))
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, epochs[i], n_batch[i], True)
        summarize_performance("faded", g_fadein, latent_dim)

        print("\n\nBurn in model_{0}".format(gen_shape))
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, epochs[i], n_batch[i])
        summarize_performance("tuned", g_normal, latent_dim)


N_BLOCKS = 6
LATENT_DIM = 128
BATCH_SIZE = [128, 128, 64, 32, 16, 8, 4]
d_models = define_discriminator(N_BLOCKS)
g_models = define_generator(LATENT_DIM, N_BLOCKS)
disc_models = define_discriminator_models(d_models, N_BLOCKS, BATCH_SIZE)
for block in range(N_BLOCKS):
    for i in [0, 1]:
        for layer in disc_models[block][i].layers:
            layer.trainable = False
        disc_models[block][i].trainable = False
        for layer in disc_models[block][i].layers:
            layer.trainable = True
        disc_models[block][i].trainable = True
gan_models = define_combined(d_models, g_models)
data_path = "data/celeba"
epochs = [10, 10, 10, 10, 10, 10]

train(g_models, disc_models, gan_models, data_path, LATENT_DIM, epochs, BATCH_SIZE)
