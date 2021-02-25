import copy
import os

import joblib
from tensorflow.keras.utils import plot_model


def save_gan(obj, path: str):
    gan = copy.copy(obj)
    f = {0, 1}
    for i in f:
        gan.combined_model[i].save_weights(os.path.join(path, f"C_{i}.h5"))
        gan.discriminator[i].save_weights(os.path.join(path, f"D_{i}.h5"))
        gan.generator[i].save_weights(os.path.join(path, f"G_{i}.h5"))


def load_progan(obj, path: str, weights_only: bool = False):
    """
    Loads the Cover GAN.

    Loads the weights of the GAN and meta information like train history and number of already trained epochs. The
    path should include a file D.h5 (weights of the discriminator), G.h5 (weights of the generator), C.5(weights of
    the combined model) and GAN.pkl (instance of the CoverGAN class)

    Args:
        obj: CoverGAN instance for which the weights are loaded
        path: Directory to the weights folder
        weights_only: Whether to save only the weights of the model

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
        obj.metrics = gan.metrics
        obj.block_images_shown = gan.block_images_shown
    return obj


def plot_progan(model, block: int, path: str, suffix: str = None):
    for i in [0, 1]:
        suffix += "_fade_in" if i == 1 else ""
        plot_model(
            model.generator[i],
            to_file=os.path.join(path, "gen{}.png".format(suffix)),
            show_shapes=True,
            expand_nested=True,
        )
        plot_model(
            model.combined_model[i],
            to_file=os.path.join(path, "comb{}.png".format(suffix)),
            show_shapes=True,
            expand_nested=True,
        )
