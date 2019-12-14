import numpy as np
from keras import backend as K
from keras.engine import Layer
from keras.layers import Add, Dense


class MinibatchStdev(Layer):

    def __init__(self, **kwargs):
        """
        Layer that adds information of the minibatch standard deviation to the output of the input tensor.
        """
        super(MinibatchStdev, self).__init__(**kwargs)

    def call(self, inputs):
        """
        @TODO
        Args:
            inputs:

        Returns:

        """
        # calculate the mean value for each pixel across channels
        mean = K.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = K.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = K.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = K.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = K.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = K.shape(inputs)
        output = K.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = K.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        """
        @TODO
        Args:
            input_shape:

        Returns:

        """
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
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

    def __init__(self, maps: int = None, use_dynamic_wscale: bool = True, **kwargs):
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
        """
        @TODO
        Args:
            inputs:

        Returns:

        """
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
