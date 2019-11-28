from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.engine import InputSpec, Layer
from keras.layers import Add


class MinibatchDiscrimination(Layer):

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        """
        Concatenates to each sample information about how different the input features for that sample are from features
        of other samples in the same minibatch, as described in Salimans et. al. (2016). Useful for preventing GANs from
        collapsing to a single output. When using this layer, generated samples and reference samples should be in
        separate batches.

        Args:

            nb_kernels: Number of discrimination kernels to use (dimensionality concatenated to output).
            kernel_dim: The dimensionality of the space where closeness of samples is calculated.
            init: name of initialization function for the weights of the layer, or alternatively, Theano function to use
                for weights initialization. This parameter is only relevant if you don't pass a `weights` argument.
            weights: list of numpy arrays to set as initial weights.
            W_regularizer: instance of WeightRegularizer (eg. L1 or L2 regularization), applied to the main weights
                matrix.
            activity_regularizer: instance of ActivityRegularizer, applied to the network output.
            W_constraint: instance of the constraints module (eg. maxnorm, nonneg), applied to the main weights matrix.
            input_dim: Number of channels/dimensions in the input. Either this argument or the keyword argument
                `input_shape`must be provided when using this layer as the first layer in a model.

        References
            - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
            - See https://github.com/forcecore/Keras-GAN-Animeface-Character/blob/master/discrimination.py for
            original source code
        """
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
                                 initializer=self.init,
                                 name='kernel',
                                 regularizer=self.W_regularizer,
                                 trainable=True,
                                 constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1] + self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output
