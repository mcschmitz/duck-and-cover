import numpy as np
from tensorflow.keras import backend as K


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


def gradient_penalty(gradients, weight):
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(
        gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))
    )
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    return K.mean(weight * K.square(1 - gradient_l2_norm))


def drift_loss(y_true, y_pred):
    return K.mean(y_pred**2)
