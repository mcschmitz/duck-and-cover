import copy
import os

import joblib
from tensorflow.keras.utils import plot_model


def save_gan(obj, path: str):
    """
    Saves the weights of the Cover GAN.

    Writes the weights of the GAN object to the given path. The combined model weights, the discriminator weights and
    the generator weights will be written separately to the given directory

    Args:
        obj: The Cover GAN object
        path: The directory to which the weights should be written
    """
    gan = copy.copy(obj)
    for block, _ in enumerate(gan.combined_model):
        for i in [0, 1]:
            gan.combined_model[block][i].save_weights(
                os.path.join(path, "C_{0}_{1}.h5".format(block, i))
            )
            gan.discriminator[block][i].save_weights(
                os.path.join(path, "D_{0}_{1}.h5".format(block, i))
            )
            gan.generator[block][i].save_weights(
                os.path.join(path, "G_{0}_{1}.h5".format(block, i))
            )

    gan.discriminator = None
    gan.generator = None
    gan.discriminator_model = None
    gan.combined_model = None
    joblib.dump(gan, os.path.join(path, "GAN.pkl"))


def load_progan(obj, path: str, weights_only: bool = False):
    """
    Loads the Cover GAN.

    Loads the weights of the GAN and meta information like train history and number of already trained epochs. The
    path should include a file D.h5 (weights of the discriminator), G.h5 (weights of the generator), C.5(weights of
    the combined model) and GAN.pkl (instance of the CoverGAN class)

    Args:
        obj: CoverGAN instance for which the weights are loaded
        path: Directory to the weights folder

    Returns:
        The CoverGAN instance with loaded weights
    """
    for block, _ in enumerate(obj.combined_model):
        for i in [0, 1]:
            obj.combined_model[block][i].load_weights(
                os.path.join(path, "C_{0}_{1}.h5".format(block, i))
            )
            obj.discriminator[block][i].load_weights(
                os.path.join(path, "D_{0}_{1}.h5".format(block, i))
            )
            obj.generator[block][i].load_weights(
                os.path.join(path, "G_{0}_{1}.h5".format(block, i))
            )
    if not weights_only:
        gan = joblib.load(os.path.join(path, "GAN.pkl"))
        obj.images_shown = gan.images_shown
        obj.history = gan.history

    return obj


def plot_progan(model, block: int, path: str, suffix: str = None):
    for i in [0, 1]:
        suffix += "_fade_in" if i == 1 else ""
        # plot_model(
        #     model.discriminator_model[block][i],
        #     to_file=os.path.join(path, "disc_m{}.png".format(suffix)),
        #     show_shapes=True,
        #     expand_nested=True,
        # )
        plot_model(
            model.generator[block][i],
            to_file=os.path.join(path, "gen{}.png".format(suffix)),
            show_shapes=True,
            expand_nested=True,
        )
        plot_model(
            model.combined_model[block][i],
            to_file=os.path.join(path, "comb{}.png".format(suffix)),
            show_shapes=True,
            expand_nested=True,
        )
