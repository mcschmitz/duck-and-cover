import keras.backend as K
import numpy as np


def wasserstein_loss(y_true, y_pred):
    """
    Calculates the Wasserstein loss for a sample batch.

    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator has a sigmoid
    output, representing the probability that samples are real or generated. In Wasserstein GANs, however,
    the output is linear with no activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the 0 and 1
    used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately. Note that
    the nature of this loss means that it can be (and frequently will be) less than 0.

    Args:
        y_true: true values for the images. Should be 1 if sample is real and -1  if sample is generated
        y_pred: predicted values for the images. Should be from a linear outout meaning [-inf, inf]

    Returns:
        Wasserstein loss for predicted and true values

    References:
        See https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py for original source code
    """
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight: float):
    """
    Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function that penalizes
    the network if the gradient norm moves away from 1. However, it is impossible to evaluate this function at all
    points in the input space. The compromise used in the paper is to choose random points on the lines between real
    and generated samples, and check the gradients at these points. Note that it is the gradient w.r.t. the input
    averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss. Then
    we get the gradients of the discriminator w.r.t. the input averaged samples. The l2 norm and penalty can then be
    calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training.

    Args:
        y_true: true y values
        y_pred: predicted y values
        averaged_samples: averaged fake and real samples
        gradient_penalty_weight: weight for the gradient penalty

    Returns:
        The average gradient penalty

    References:
        See https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py for original source code
    """

    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)
