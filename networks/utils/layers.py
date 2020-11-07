import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Conv2D, Dense, Layer
from tensorflow.python.ops.init_ops_v2 import _compute_fans


class MinibatchSd(Layer):
    """
    Calculates the minibatch standard deviation and adds it to the output.

    Calculates the average standard deviation of each value in the input
    map across the channels and concatenates a blown up version of it
    (same size as the input map) to the input
    """

    def call(self, inputs):
        mean = K.mean(inputs, axis=0, keepdims=True)
        squared_diff = K.square(inputs - mean)
        avg_squared_diff = K.mean(squared_diff, axis=0, keepdims=True)
        avg_squared_diff += 1e-8
        sd = K.sqrt(avg_squared_diff)
        avg_sd = K.mean(sd, keepdims=True)
        shape = K.shape(inputs)
        output = K.tile(avg_sd, (shape[0], shape[1], shape[2], 1))
        combined = K.concatenate([inputs, output], axis=-1)
        return combined

    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)


class RandomWeightedAverage(Add):
    """
    Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest
    solution I could think of. Improvements appreciated

    References:
       See https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py for original source code
    """

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError("inputs has to be of length 2.")
        batch_size = tf.shape(inputs[0])[0]
        weights = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        return weights * inputs[0] + (1 - weights) * inputs[1]


class WeightedSum(Add):
    def __init__(self, alpha: float = 0.0, **kwargs):
        """
        Weighted sum layer that outputs the weighted sum of two input tensors.

        Calculates the weighted sum a * (1 - alpha) + b * (alpha) for the input
        tensors a and b and weight alpha

        Args:
            alpha: alpha weight
            **kwargs: keyword args passed to parent class
        """
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name="ws_alpha")

    def _merge_function(self, inputs):
        assert len(inputs) == 2
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class ScaledDense(Dense):
    def __init__(
        self,
        gain: float = None,
        use_dynamic_wscale: bool = True,
        **kwargs,
    ):
        """
        Dense layer with weight scaling.

        Ordinary Dense layer, that scales the weights by He's scaling
        factor at each forward pass. He's scaling factor is defined as
        sqrt(2/fan_in) where fan_in is the number of input units of the
        layer
        """
        super(ScaledDense, self).__init__(**kwargs)
        self.use_dynamic_wscale = use_dynamic_wscale
        self.gain = gain if gain else np.sqrt(2)

    def call(self, inputs):
        fan_in = float(_compute_fans(self.weights[0].shape)[0])
        wscale = self.gain / np.sqrt(max(1.0, fan_in))
        output = tf.tensordot(inputs, self.kernel * wscale, 1)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format="channels_last")
        if self.activation is not None:
            output = self.activation(output)
        return output


class ScaledConv2D(Conv2D):
    def __init__(
        self,
        gain: float = np.sqrt(2),
        use_dynamic_wscale: bool = True,
        **kwargs,
    ):
        """
        Scaled Convolutional Layer.

        Scales the weights on each forward pass down to by He's initialization
        factor. For Progressive Growing GANS this results in an equalized
        learning rate for all layers, where bigger layers are scaled down.

        Args:
            gain: Constant to upscale the weight sclaing
            use_dynamic_wscale: Switch on or off dynamic weight scaling.
                Switching it off results in a ordinary 2D convolutional layer.
            **kwargs:
        """
        super(ScaledConv2D, self).__init__(**kwargs)
        self.use_dynamic_wscale = use_dynamic_wscale
        self.gain = gain

    def call(self, inputs):
        fan_in = float(_compute_fans(self.weights[0].shape)[0])
        wscale = self.gain / np.sqrt(max(1.0, fan_in))
        outputs = K.conv2d(
            inputs,
            self.kernel * wscale,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )

        if self.use_bias:
            outputs = K.bias_add(
                outputs, self.bias, data_format=self.data_format
            )

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class PixelNorm(Layer):
    def __init__(self, **kwargs):
        """
        Pixel normalization layer that normalizes an input tensor along its
        channel axis by scaling its features along this axis to unit length.

        Args:
            channel_axis: channel axis. tensor value will be normalized along
                this axis
            kwargs: Keyword arguments for keras layer
        """
        super(PixelNorm, self).__init__(**kwargs)

    def call(self, inputs):
        values = inputs ** 2.0
        mean_values = K.mean(values, axis=-1, keepdims=True)
        mean_values += 1.0e-8
        l2 = K.sqrt(mean_values)
        normalized = inputs / l2
        return normalized

    def compute_output_shape(self, input_shape):
        return input_shape


class GradientPenalty(Layer):
    def __init__(self, weight: int = 1):
        """
        Gradient Penalty Layer.

        Computes the gradient penalty for a given set of targets and gradients.

        Args:
            weight: Allows weighting of the Gradient Penalty
        """
        super(GradientPenalty, self).__init__()
        self.weight = weight

    def call(self, inputs):
        (target, wrt) = inputs
        gradients = K.gradients(target, wrt)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(
            gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
        )
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = self.weight * K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)
