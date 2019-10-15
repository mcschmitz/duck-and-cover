import copy
import os

import joblib
import tensorflow as tf
from keras.layers import Layer
from keras.models import load_model


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


def save_gan(obj, path):
    """
    @TODO
    Args:
        gan:
        path:

    Returns:

    """
    gan = copy.copy(obj)
    gan.discriminator.trainable = False
    gan.combined_model.save(os.path.join(path, "C.h5"))
    gan.discriminator.trainable = True
    gan.discriminator.save(os.path.join(path, "D.h5"))
    gan.generator.save(os.path.join(path, "G.h5"))

    gan.discriminator = None
    gan.generator = None
    gan.discriminator_model = None
    gan.combined_model = None
    joblib.dump(gan, os.path.join(path, "GAN.pkl"))


def load_cover_gan(path, custom_objects: dict = None):
    """
    @TODO
    Returns:

    """
    discriminator = load_model(os.path.join(path, "D.h5"), custom_objects=custom_objects)
    generator = load_model(os.path.join(path, "G.h5"), custom_objects=custom_objects)
    generator.trainable = False
    combined_model = load_model(os.path.join(path, "C.h5"), custom_objects=custom_objects)
    gan = joblib.load(os.path.join(path, "GAN.pkl"))
    gan.generator = generator
    gan.discriminator = discriminator
    gan.combined_model = combined_model

    return gan
