import os
from typing import List

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from utils import logger


class DataLoader(Sequence):
    def __init__(
        self,
        image_path: str,
        image_size: int = 256,
        binarizer: MultiLabelBinarizer = None,
        batch_size: int = 32,
    ):
        """
        Image loader that takes a path and crawls the directory and
        subdirectories for images and loads them.

        Args:
            image_path: Path to the dictionary that should be crawled for image data
            image_size: output size of the images
            binarizer: Binarizer used to create dummy features out of additional meta information
            batch_size: Size of one Batch
        """
        self.binarizer = binarizer
        self._iterator_i = 0
        self.image_size = image_size
        self.batch_size = batch_size

        np_path = os.path.join(image_path, f"all{image_size}.npy")
        if os.path.exists(np_path):
            try:
                self._images = np.load(np_path)
            except MemoryError:
                logger.warning(
                    "Data does not fit into memory. Will be streamed from disk"
                )
                self._images = get_image_paths(image_path)
                self._iterator = np.arange(0, self._images.shape[0])
        else:
            try:
                files = get_image_paths(image_path)
                self._images = np.zeros(
                    (len(files), 3, image_size, image_size)
                )

                for i, file_path in enumerate(tqdm(files)):
                    img = imread(file_path)
                    img = resize(img, (image_size, image_size, 3))
                    img = np.moveaxis(img, -1, 0)
                    self._images[i] = img
                np.save(np_path, self._images)
                self._iterator = np.arange(0, self._images.shape[0])
            except MemoryError:
                logger.warning(
                    "Data does not fit into memory. Will be streamed from disk"
                )
                self._images = get_image_paths(image_path)

        self.n_images = len(self._images)

    def __len__(self):
        return self.n_images // self.batch_size

    def __getitem__(self, item):
        batch_x = np.zeros(
            (self.batch_size, 3, self.image_size, self.image_size)
        )
        batch_idx = self._get_batch_idx()
        if isinstance(self._images, np.ndarray):
            batch_x = self._images[batch_idx]
        else:
            for i, b_idx in enumerate(batch_idx):
                file_path = self._images[b_idx]
                try:
                    img = imread(file_path)
                    img = np.moveaxis(img, -1, 0)
                except ValueError as err:
                    logger.error(f"Unable to load {file_path}. Error: {err}")
                img = resize(img, (3, self.image_size, self.image_size))
                batch_x[i] = img
        self._iterator_i = batch_idx[-1]
        return DataLoader._wrap_output(batch_x)

    def _get_batch_idx(self):
        positions = np.arange(
            self._iterator_i, self._iterator_i + self.batch_size
        )
        batch_idx = [
            i if i < self.n_images else i - self.n_images for i in positions
        ]
        if 0 in batch_idx:
            logger.info("Data Generator exceeded. Will shuffle input data.")
            np.random.shuffle(self._images)
        return batch_idx

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
        jpgs = {f for f in filenames if f.endswith(".jpg")}
        for filename in jpgs:
            paths.append(os.path.join(dirpath, filename))
    return paths
