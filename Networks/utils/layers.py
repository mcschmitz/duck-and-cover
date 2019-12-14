import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.layers import Add, Dense, Conv2D


class MinibatchSd(Layer):

    def __init__(self, **kwargs):
        """Calculates the minibatch standard deviation and adds it to the output

        Calculates the average standard deviation of each value in the input map across the channels and
        concatenates a blown up version of it (same size as the input map) to the input
        """
        super(MinibatchSd, self).__init__(**kwargs)

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


class WeightedSum(Add):

    def __init__(self, alpha=0.0, **kwargs):
        """
        @TODO
        Args:
            alpha:
            **kwargs:
        """
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    def _merge_function(self, inputs):
        """
        @TODO
        Args:
            inputs:

        Returns:

        """
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class ScaledDense(Dense):

    def __init__(self, maps=None, use_dynamic_wscale: bool = True, **kwargs):
        """
        @TODO
        Args:
            maps:
            use_dynamic_wscale:
            **kwargs:
        """
        super(ScaledDense, self).__init__(**kwargs)
        self.use_dynamic_wscale = use_dynamic_wscale
        self.maps = maps

    def call(self, inputs):
        if self.maps is not None:
            fan_in = np.prod(self.maps)
        else:
            fan_in = np.prod(inputs.shape[1:]).value
        wscale = np.sqrt(2) / np.sqrt(fan_in)
        output = K.dot(inputs, self.kernel * wscale)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class ScaledConv2D(Conv2D):

    def __init__(self, maps: int = None, use_dynamic_wscale: bool = True, **kwargs):
        """
        @TODO
        Args:
            maps:
            use_dynamic_wscale:
            **kwargs:
        """
        super(ScaledConv2D, self).__init__(**kwargs)
        self.use_dynamic_wscale = use_dynamic_wscale
        self.maps = maps

    def call(self, inputs):
        if self.maps is not None:
            fan_in = np.prod(self.maps)
        else:
            fan_in = np.prod(inputs.shape[1:]).value
        wscale = np.sqrt(2) / np.sqrt(fan_in)

        outputs = K.conv2d(inputs, self.kernel * wscale, strides=self.strides, padding=self.padding,
                           data_format=self.data_format, dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
