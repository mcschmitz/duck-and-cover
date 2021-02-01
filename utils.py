import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

plt.ioff()


def scale_images(images: np.array):
    """
    Scales the images to [-1, 1]

    Args:
        images: numpy array of images

    Returns:
        The scaled images as numpy array
    """
    images = (images - 127.5) / 127.5
    return images


def resize_images(images: np.array, new_shape: tuple):
    """
    Scales images to desired size.

    Args:
        images: List of images
        new_shape: Tuple of the desired image size
    """
    images_list = list()
    for image in tqdm(images):
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)
