import os

import numpy as np
from PIL import Image

from Loader import rescale_images


def create_dir(directory_path):
    """
    Creates a directory to the given path if it does not exist and returns the path

    Args:
        directory_path: path directory

    Returns:
        the created path
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path


def generate_images(generator_model, output_dir, year: float = None, n_imgs: int = 10, fixed: bool = False):
    """TODO
    Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    if fixed:
        np.random.seed(101)
    else:
        np.random.seed()
    if year is None:
        generated_images = generator_model.predict(np.random.normal(size=(n_imgs, generator_model.input_shape[1])))
    else:
        year = np.repeat(year, n_imgs)
        noise = np.random.normal(size=(n_imgs, generator_model.input_shape[0][1]))
        generated_images = generator_model.predict([noise, year])
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
