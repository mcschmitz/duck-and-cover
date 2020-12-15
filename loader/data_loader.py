import logging
import os
from typing import List

import numpy as np
import psutil
import tensorflow.keras.backend as K  # noqa: WPS301
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from constants import LOG_DATETIME_FORMAT, LOG_FORMAT, LOG_LEVEL

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger(__file__)


class DataLoader(object):
    def __init__(
        self,
        image_path: str,
        image_size: int = 256,
        binarizer: MultiLabelBinarizer = None,
    ):
        """
        Image loader that takes a path and crawls the directory and
        subdirectories for images and loads them.

        Args:
            image_path: Path to the dictionary that should be crawled for image data
            image_size: output size of the images
            binarizer: Binarizer used to create dummy features out of additional meta information
        """
        self.binarizer = binarizer
        self._iterator_i = 0
        self.image_size = image_size

        np_path = os.path.join(image_path, f"all{image_size}.npy")
        data_loadable = os.stat(np_path).st_size < (
            psutil.virtual_memory().total * 0.8
        )
        if os.path.exists(np_path) and data_loadable:
            self._images = np.load(np_path)
            self._iterator = np.arange(0, self._images.shape[0])
        elif os.path.exists(np_path):
            logger.info(f"Load data from {np_path}")
            self._images = get_image_paths(image_path)
        else:
            try:
                files = get_image_paths(image_path)
                self._images = np.zeros(
                    (len(files), image_size, image_size, 3), dtype=K.floatx()
                )

                for i, file_path in enumerate(tqdm(files)):
                    img = imread(file_path)
                    img = resize(img, (image_size, image_size, 3))
                    self._images[i] = img
                np.save(np_path, self._images)
                self._iterator = np.arange(0, self._images.shape[0])
            except MemoryError:
                logger.warning(
                    "Data does not fit into memory. Data will be streamed from disk"
                )
                self._images = get_image_paths(image_path)

        self.n_images = len(self._images)

    def get_next_batch(self, batch_size: int = 32):
        """
        Loads the next batch of images.

        Args:
            batch_size: size of the drawn batch

        Returns:
            List of return values. Contains numpy array of images and release year information as well as genre
                information if requested
        """
        batch_x = np.zeros(
            (batch_size, self.image_size, self.image_size, 3), dtype=K.floatx()
        )
        batch_idx = [
            i if i < self.n_images else i - self.n_images
            for i in np.arange(self._iterator_i, self._iterator_i + batch_size)
        ]
        if isinstance(self._images, np.ndarray):
            batch_x = self._images[batch_idx]
        else:
            for i, b_idx in enumerate(batch_idx):
                file_path = self._images[b_idx]
                try:
                    img = imread(file_path)
                except ValueError as err:
                    logger.error(f"Unable to load {file_path}. Error: {err}")
                img = resize(img, (self.image_size, self.image_size, 3))
                batch_x[i] = img
        if 0 in batch_idx:
            np.random.shuffle(self._images)
        self._iterator_i = batch_idx[-1]
        return DataLoader._wrap_output(batch_x)

    @classmethod
    def _wrap_output(
        cls, x: np.array = None, genres: np.array = None, year: np.array = None
    ):
        """
        Wraps the output of the data loader to one object.

        Returns the images and if required the genre and year information

        Args:
            x: numpy array of images
            genres: numpy array of genre information
            year: numpy array of release year information

        Returns:
            List of return values. Contains numpy array of images and release year information as well as genre
                information if requested
        """
        return_values = [x]
        if year is not None:
            return_values.append(year)
        if genres is not None:
            return_values.append([genres])
        if len(return_values) == 1:
            return return_values[0]
        return return_values


def get_image_paths(directory: str) -> List[str]:
    """
    Crawls a directory for images.

    Args:
        directory: Directory to crawl
    """
    paths = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        jpgs = [f for f in filenames if f.endswith(".jpg")]
        for filename in jpgs:
            paths.append(os.path.join(dirpath, filename))
    return paths
