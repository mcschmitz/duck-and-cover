import tensorflow as tf
from keras.engine import Layer


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
