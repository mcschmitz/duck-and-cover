import os

import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import array_to_img

plt.ioff()


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


def generate_images(generator_model, output_dir, n_imgs: int = 10, seed: int = None, target_size: tuple = (64, 64)):
    """Generates a list of images by predicting with the given generator

    Feeds normal distributed random numbers into the generator to generate `n_imgs`, tiles the image, rescales it and
    and saves the output to a PNG file.

    Args:
        generator_model: Generator used for generating the images
        output_dir: where to save the results
        n_imgs: number of images to generate
        seed: seed to use to generate the numbers of the latent space
        target_size: target size of the image
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    generated_images = generator_model.predict(np.random.normal(size=(n_imgs, generator_model.input_shape[1])))
    # generated_images = rescale_images(generated_images)
    tiled_output = tile_images(generated_images)
    tiled_output = array_to_img(tiled_output, scale=True)
    tiled_output = tiled_output.resize(size=target_size)
    tiled_output.save(output_dir)


def tile_images(image_stack):
    """TODO
    Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class AnimatedGif:

    def __init__(self, size: tuple = (640, 480)):
        """
        @TODO + Refs https://tomroelandts.com/articles/how-to-create-animated-gifs-with-python

        Args:
            size:
        """
        self.size = size
        self.fig = plt.figure()
        self.fig.set_size_inches(self.size[0] / 100, self.size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []

    def add(self, image, label='', label_position: tuple = (1, 1)):
        plt_im = plt.imshow(image, vmin=0, vmax=1, animated=True)
        plt_txt = plt.text(label_position[0], label_position[1], label, color='black')
        self.images.append([plt_im, plt_txt])

    def save(self, filename, fps: float = 10):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=fps)
