import keras.backend as K
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import os
from utils import scale_images
import psutil
from skimage.io import imread
from skimage.transform import resize


class DataLoader(object):
    def __init__(
        self, image_path: str, image_size: int = 256, binarizer: MultiLabelBinarizer = None,
    ):
        """
        Image Loader that takes a path and crawls the directory and
        subdirectories for images and loads them.

        Args:
            image_path: Path to the dictionary that should be crawled for image data
            image_size: output size of the images
            binarizer: Binarizer used to create dummy features out of additional meta information
        """
        self.binarizer = binarizer
        self._iterator_i = 0
        self.image_size = image_size

        np_path = os.path.join(image_path, "all{0}.npy".format(image_size))
        if os.path.exists(np_path) and os.stat(np_path).st_size < (psutil.virtual_memory().total * 0.8):
            self._images = np.load(np_path)
            self._iterator = np.arange(0, self._images.shape[0])
        elif os.path.exists(np_path):
            self._images = []
            for dirpath, dirnames, filenames in os.walk(image_path):
                for filename in [f for f in filenames if f.endswith(".jpg")]:
                    self._images.append(os.path.join(dirpath, filename))
        else:
            try:
                files = []
                for dirpath, dirnames, filenames in os.walk(image_path):
                    for filename in [f for f in filenames if f.endswith(".jpg")]:
                        files.append(os.path.join(dirpath, filename))
                self._images = np.zeros((len(files), image_size, image_size, 3), dtype=K.floatx())

                for i, file_path in tqdm(enumerate(files)):
                    img = imread(file_path)
                    img = resize(img, (image_size, image_size, 3))
                    img = scale_images(img)
                    self._images[i] = img
                np.save(np_path, self._images)
                self._iterator = np.arange(0, self._images.shape[0])
            except MemoryError:
                self._images = []
                for dirpath, dirnames, filenames in os.walk(image_path):
                    for filename in [f for f in filenames if f.endswith(".jpg")]:
                        self._images.append(os.path.join(dirpath, filename))

        self._n_images = self._images.shape[0]

    def next(self, batch_size: int = 32):
        """
        Loads the next batch of images.

        Args:
            batch_size: size of the drawn batch

        Returns:
            List of return values. Contains numpy array of images and release year information as well as genre
                information if requested
        """
        batch_x = np.zeros((batch_size, self.image_size, self.image_size, 3), dtype=K.floatx())

        if isinstance(self._images, np.array):
            batch_idx = [
                i if i < self._n_images else i - self._n_images
                for i in np.arange(self._iterator_i, self._iterator_i + batch_size)
            ]
            batch_x = self._images[batch_idx]
        else:
            batch_idx = [
                i if i < self._n_images else i - self._n_images
                for i in np.arange(self._iterator_i, self._iterator_i + batch_size)
            ]
            for i, b_idx in enumerate(batch_idx):
                file_path = self._images[b_idx]
                img = imread(file_path)
                img = resize(img, (self.image_size, self.image_size, 3))
                batch_x[i] = scale_images(img)
        if 0 in batch_idx:
            np.random.shuffle(self._images)
        return self._wrap_output(batch_x)

    @staticmethod
    def _wrap_output(x: np.array = None, genres: np.array = None, year: np.array = None):
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
