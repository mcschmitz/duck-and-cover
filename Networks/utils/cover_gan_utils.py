import copy
import os

import joblib
import tensorflow as tf
from keras.layers import Layer


def _pixel_norm(x, epsilon=1e-8, channel_axis=-1):
    """
    Normalizes the features for each pixel across its channels to unit length

    Args:
        x: the input tensor x
        epsilon: small value to prevent division by zero
        channel_axis: channel axis. Pixel value will be normalized along this axis

    Returns:
        The normalized tensor
    """
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=channel_axis, keepdims=True) + epsilon)


class PixelNorm(Layer):

    def __init__(self, channel_axis=-1, **kwargs):
        """
        Pixel normalization layer that normalizes an innput tensor along its channel axis by scaling its features
        along this axis to unit length

        Args:
            channel_axis: channel axis. tensor value will be normalized along this axis
            kwargs: Keyword arguments for keras layer
        """
        self.channel_axis = channel_axis
        super().__init__()

    def call(self, x):
        return _pixel_norm(x, channel_axis=self.channel_axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {
            'channel_axis': self.channel_axis,
            **super().get_config()
        }


def save_gan(obj, path: str):
    """
    Saves the weights of the Cover GAN

    Writes the weights of the GAN object to the given path. The combined model weights, the discriminator weights and
    the generator weights will be written separately to the given directory

    Args:
        obj: The Cover GAN object
        path: The directory to which the weights should be written
    """
    gan = copy.copy(obj)
    gan.combined_model.save_weights(os.path.join(path, "C.h5"))
    gan.discriminator.save_weights(os.path.join(path, "D.h5"))
    gan.generator.save_weights(os.path.join(path, "G.h5"))

    gan.discriminator = None
    gan.generator = None
    gan.discriminator_model = None
    gan.combined_model = None
    joblib.dump(gan, os.path.join(path, "GAN.pkl"))


def load_cover_gan(obj, path: str):
    """
    Loads the weights of the Cover GAN

    Loads the weights of the GAN and meta information like train history and number of already trained epochs. The
    path should include a file D.h5 (weights of the discriminator), G.h5 (weights of the generator), C.5(weights of
    the combined model) and GAN.pkl (instance of the CoverGAN class)

    Args:
        obj: CoverGAN instance for which the weights are loaded
        path: Directory to the weights folder

    Returns:
        The CoverGAN instance with loaded weights
    """
    obj.discriminator.load_weights(os.path.join(path, "D.h5"))
    obj.generator.load_weights(os.path.join(path, "G.h5"))
    obj.combined_model.load_weights(os.path.join(path, "C.h5"))
    gan = joblib.load(os.path.join(path, "GAN.pkl"))
    obj.n_epochs = gan.n_epochs
    obj.history = gan.history

    print("Generator summary:\n")
    print(obj.generator.summary())
    print("Discriminator summary:\n")
    print(obj.discriminator.summary())
    print("Combined model summary:\n")
    print(obj.combined_model.summary())

    return obj
