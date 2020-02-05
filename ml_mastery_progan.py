"""
This script defines and trains the ProGAN as defined in
https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-
keras-for-synthesizing-faces/, whereby some functions and classes are modified
or enhanced.
"""

import os
from math import sqrt

import numpy as np
import psutil
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import AveragePooling2D, Flatten, Input, LeakyReLU, Reshape, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from matplotlib import pyplot
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from networks.utils import PixelNorm, MinibatchSd, WeightedSum, ScaledConv2D, ScaledDense, wasserstein_loss


def add_discriminator_block(old_model: Model, n_input_layers: int = 3) -> list:
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

    d = ScaledConv2D(128, kernel_size=(1, 1), padding="same", kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D(2, 2)(d)
    block_new = d

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    model1 = Model(in_image, d)
    model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    downsample = AveragePooling2D(2, 2)(in_image)
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    d = WeightedSum()([block_old, block_new])

    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    model2 = Model(in_image, d)
    model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]


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

    d = ScaledConv2D(128, kernel_size=(1, 1), padding="same", kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = MinibatchSd()(d)
    d = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ScaledConv2D(128, kernel_size=(4, 4), padding="same", kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Flatten()(d)
    out_class = ScaledDense(1, gain=1)(d)

    model = Model(in_image, out_class)
    model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    model_list.append([model, model])
    for i in range(1, n_double):
        old_model = model_list[i - 1][0]
        models = add_discriminator_block(old_model)
        model_list.append(models)
    return model_list


def add_generator_block(old_model: Model) -> list:
    """
    Adds a new block to the generator model.

    Args:
        old_model: Already well trained generator model

    Returns:
        list of old and new model
    """
    init = RandomNormal(0, 1)

    block_end = old_model.layers[-2].output
    upsampling = UpSampling2D()(block_end)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(upsampling)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(g)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = ScaledConv2D(3, kernel_size=(1, 1), padding="same", kernel_initializer=init)(g)
    model1 = Model(old_model.input, out_image)

    out_old = old_model.layers[-1]
    out_image2 = out_old(upsampling)
    merged = WeightedSum()([out_image2, out_image])

    model2 = Model(old_model.input, merged)
    return [model1, model2]


def define_generator(latent_dim: int, n_doublings: int, img_dim: int = 4) -> list:
    """
    Defines a list of generator models.

    Args:
        latent_dim: Dimension of the latent space
        n_doublings: Number of doublings of the image resolution
        img_dim: image input dimensoon

    Returns:
    """
    init = RandomNormal(0, 1)
    model_list = list()

    in_latent = Input(shape=(latent_dim,))
    g = ScaledDense(128 * img_dim * img_dim, kernel_initializer=init, gain=np.sqrt(2) / 4)(in_latent)
    g = Reshape((img_dim, img_dim, 128))(g)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(g)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding="same", kernel_initializer=init)(g)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = ScaledConv2D(3, kernel_size=(1, 1), padding="same", kernel_initializer=init, gain=1)(g)

    model = Model(in_latent, out_image)
    model_list.append([model, model])

    for i in range(1, n_doublings):
        old_model = model_list[i - 1][0]
        models = add_generator_block(old_model)
        model_list.append(models)
    return model_list


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
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        model_list.append([model1, model2])
    return model_list


def load_real_samples(path: str, size: int = 4):
    """
    Loads the image dataset.

    Args:
        path: path to the image files
        size: target resolution of the image tensor

    Returns:
        list of image tensor and image index
    """
    if os.path.exists(path) and os.stat(path).st_size < (psutil.virtual_memory().total * 0.8):
        images = np.load(path)
        img_idx = np.arange(0, images.shape[0])
        return images, img_idx
    elif os.path.exists(path):
        print("Data does not fit inside Memory. Preallocation is not possible, use iterator instead")
    else:
        try:
            files = [
                os.path.join(path, f) for f in os.listdir(path) if os.path.splitext(os.path.join(path, f))[1] == ".jpg"
            ]
            images = np.zeros((len(files), size, size, 3), dtype=K.floatx())

            for i, file_path in tqdm(enumerate(files)):
                x = imread(file_path)
                x = resize(x, (size, size, 3))
                x = (x - 127.5) / 127.5
                images[i] = x
            np.save(path, images)
            img_idx = np.arange(0, images.shape[0])
            return images, img_idx
        except MemoryError as _:
            print("Data does not fit inside Memory. Preallocation is not possible.")


def generate_real_samples(dataset, n_samples):
    """
    Selects n random images out of the dataset.

    Args:
        dataset: list-like dataset
        n_samples: how many sample to draw

    Returns:
        sampled datapoints and y values
    """
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y


def generate_latent_points(latent_dim, n_samples):
    """
    Samples random points.

    Args:
        latent_dim: dimension of the latent space
        n_samples: number of samples to draw

    Returns:
        sampled points
    """
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def generate_fake_samples(generator: Model, latent_dim: int, n_samples: int):
    """
    Use the generator to generate fake images.

    Args:
        generator: Generator model to use
        latent_dim: dimension of the latent space
        n_samples: number of samples to generate

    Returns:
        generated samples and their class
    """
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = -np.ones((n_samples, 1))
    return X, y


def update_fadein(models: list, step: int, n_steps: int):
    """
    Updates the alpha values of the fade in layers

    Args:
        models: list of models to update
        step: current training step
        n_steps: overall number of training steps
    """
    alpha = step / float(n_steps - 1)
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)


def train_epochs(g_model: Model, d_model: Model, gan_model: Model, dataset: np.array, n_epochs: int, batch_size: int,
                 fadein: bool = False):
    """
    Trains the model

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
    half_batch = batch_size // 2
    for i in range(n_steps):
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        x_real, y_real = generate_real_samples(dataset, half_batch)
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss1 = d_model.train_on_batch(x_real, y_real)
        d_loss2 = d_model.train_on_batch(x_fake, y_fake)
        z_input = generate_latent_points(latent_dim, batch_size)
        y_real2 = np.ones((batch_size, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        print(">%d, d1=%.3f, d2=%.3f g=%.3f" % (i + 1, d_loss1, d_loss2, g_loss))


def scale_dataset(images: list, new_shape: tuple):
    """
    Scales images to desired size.

    Args:
        images: List of images
        new_shape: Tuple of the desired image size
    """
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)


# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
    # devise name
    gen_shape = g_model.output_shape
    name = "%03dx%03d-%s" % (gen_shape[1], gen_shape[2], status)
    # generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis("off")
        pyplot.imshow(X[i])
    # save plot to file
    filename1 = "plot_%s.png" % (name)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = "model_%s.h5" % (name)
    g_model.save(filename2)
    print(">Saved: %s and %s" % (filename1, filename2))


# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    # fit the baseline model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print("Scaled Data", scaled_data.shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance("tuned", g_normal, latent_dim)
    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print("Scaled Data", scaled_data.shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
        summarize_performance("faded", g_fadein, latent_dim)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance("tuned", g_normal, latent_dim)


# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 6
# size of the latent space
latent_dim = 100
# define models
d_models = define_discriminator(n_blocks)
# define models
g_models = define_generator(latent_dim, n_blocks)
# define composite models
gan_models = define_combined(d_models, g_models)
# load image data
dataset = load_real_samples("img_align_celeba_128.npz")
print("Loaded", dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)
