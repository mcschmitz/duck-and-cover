"""
This script defines and trains the ProGAN as defined in
https://machinelearningmastery.com/how-to-train-a-progressive-growing-gan-in-keras-for-synthesizing-faces/,
whereby some functions and classes are modified or enhanced.
"""

import os
from math import sqrt

import numpy as np
import psutil
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot
from numpy import asarray
from numpy import ones
from numpy.random import randint
from numpy.random import randn
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

from networks.utils import PixelNorm, MinibatchSd, WeightedSum, ScaledConv2D, ScaledDense, wasserstein_loss


def add_discriminator_block(old_model: Model, n_input_layers: int = 3) -> list:
    """
    Adds a new block to the discriminator model

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

    d = ScaledConv2D(128, kernel_size=(1, 1), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(d)
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
    Defines a list of discriminator models

    Args:
        n_double: Number of doublings of the image resolution
        input_shape: Shape of the input image

    Returns:
        list of discriminator models
    """
    init = RandomNormal(0, 1)
    model_list = list()

    in_image = Input(shape=input_shape)

    d = ScaledConv2D(128, kernel_size=(1, 1), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    d = MinibatchSd()(d)
    d = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ScaledConv2D(128, kernel_size=(4, 4), padding='same', kernel_initializer=init)(d)
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
    Adds a new block to the generator model

    Args:
        old_model: Already well trained generator model

    Returns:
        list of old and new model
    """
    init = RandomNormal(0, 1)

    block_end = old_model.layers[-2].output
    upsampling = UpSampling2D()(block_end)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(upsampling)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(g)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = ScaledConv2D(3, kernel_size=(1, 1), padding='same', kernel_initializer=init)(g)
    model1 = Model(old_model.input, out_image)

    out_old = old_model.layers[-1]
    out_image2 = out_old(upsampling)
    merged = WeightedSum()([out_image2, out_image])

    model2 = Model(old_model.input, merged)
    return [model1, model2]


def define_generator(latent_dim: int, n_doublings: int, img_dim: int = 4) -> list:
    """
    Defines a list of generator models

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
    g = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(g)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = ScaledConv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer=init)(g)
    g = PixelNorm()(g)
    g = LeakyReLU(alpha=0.2)(g)
    out_image = ScaledConv2D(3, kernel_size=(1, 1), padding='same', kernel_initializer=init, gain=1)(g)

    model = Model(in_latent, out_image)
    model_list.append([model, model])

    for i in range(1, n_doublings):
        old_model = model_list[i - 1][0]
        models = add_generator_block(old_model)
        model_list.append(models)
    return model_list


def define_combined(discriminators: list, generators: list) -> list:
    """
    Build the combined models out of the given generator and discriminator models.

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
    Loads the image dataset

    Args:
        path: path to the image files
        size: target resolution of the image tensor

    Returns:
        list of image tensor and image index
    """
    if os.path.exists(path) and os.stat(path).st_size < (psutil.virtual_memory().total * .8):
        images = np.load(path)
        img_idx = np.arange(0, images.shape[0])
        return images, img_idx
    elif os.path.exists(path):
        print("Data does not fit inside Memory. Preallocation is not possible, use iterator instead")
    else:
        try:
            files = [os.path.join(path, f) for f in os.listdir(path) if
                     os.path.splitext(os.path.join(path, f))[1] == ".jpg"]
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


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # select images
    X = dataset[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -ones((n_samples, 1))
    return X, y


# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)


# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # update alpha for all WeightedSum layers when fading in new blocks
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        # prepare real and fake samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update the generator via the discriminator's error
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, d_loss1, d_loss2, g_loss))


# scale images to preferred size
def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
    # devise name
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
    # generate images
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i])
    # save plot to file
    filename1 = 'plot_%s.png' % (name)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%s.h5' % (name)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    # fit the baseline model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance('tuned', g_normal, latent_dim)
    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print('Scaled Data', scaled_data.shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
        summarize_performance('faded', g_fadein, latent_dim)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance('tuned', g_normal, latent_dim)


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
dataset = load_real_samples('img_align_celeba_128.npz')
print('Loaded', dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)
