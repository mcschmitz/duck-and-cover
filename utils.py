import os

import numpy as np
from PIL import Image

from Loader import rescale_images


def create_dir(directory_path):
    """
    Creates a directory to the given path if it does not exist
    Args:
        directory_path: path directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def generate_images(generator_model, output_dir, n_imgs: int = 10):
    """TODO
    Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    generated_images = generator_model.predict(np.random.rand(n_imgs, generator_model.input_shape[1]))
    generated_images = rescale_images(generated_images)
    tiled_output = tile_images(generated_images)
    tiled_output = Image.fromarray(tiled_output)
    tiled_output.save(output_dir)


def tile_images(image_stack):
    """TODO
    Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images
