import tensorflow as tf
from keras.layers import Layer


def _pixel_norm(x, epsilon=1e-8, channel_axis=-1):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=channel_axis, keepdims=True) + epsilon)


class PixelNorm(Layer):
    def __init__(self, channel_axis=-1, **kwargs):
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
