"""
Fairly basic set of tools for real-time data augmentation on image data. Can easily be extended to include new
transformations, new preprocessing methods, etc...

Some of the  functions are from https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py
"""

import keras.backend as K
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image as pil_image
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


def random_channel_shift(x: np.array, intensity: float, channel_axis: int = 0):
    """
    Randomly shifts color channels of the image array.

    Args:
        x: input image
        intensity: intensity of the channel shift
        channel_axis: axis on which along the shift should be executed. Usually the color channel axis

    Returns:
        The augmented image

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix: np.array, x: int, y: int):
    """
    Offsets an input matrix from its center.

    Takes an input matrix M that is used for performing geometric transformation on matrix A and offsets it from its
    center.

    Args:
        matrix: the input matrix
        x: integer giving the height of matrix A
        y: integer giving the width of matrix A

    Returns:
        The offset input matrix

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x: np.array, transform_matrix: np.array, channel_axis: int = 0, fill_mode: str = 'nearest',
                    cval: float = 0.):
    """
    Apply the image transformation specified by a transformation matrix.

    Args:
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input are filled according to the given mode (one of `{
            'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries of the input if `mode='constant'`.

    Returns:
        The transformed version of the input.

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x: np.array, axis: int):
    """
    Flips the axis of a given input matrix

    Args:
        x: 2D numpy array, single image.
        axis: Index of the axis to flip

    Returns:
        The flipped input array

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x: np.array, data_format: str = None, scale: bool = True):
    """
    Converts a 3D Numpy array to a PIL Image instance.

    Args:
        x: Input Numpy array.
        data_format: Image data format. Can be `channels_first` or `channels_last` indicating whether the channel
            axis of the image should be the first or the last axis.
        scale: Whether to rescale image values
            to be within [0, 255].

    Returns
        A PIL Image instance.

    Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=K.floatx())
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)

    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def img_to_array(img, data_format: str = None):
    """
    Converts a PIL Image instance to a Numpy array.

    Args:
        img: PIL Image instance.
        data_format: Image data format. Can be `channels_first` or `channels_last` indicating whether the channel
            axis of the image should be the first or the last axis.

    Returns:
        A 3D Numpy array.

    Raises
        ValueError: if invalid `img` or `data_format` is passed.

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=K.floatx())
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


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


def rescale_images(images: np.array):
    """
    Scales the images back from a range of [-1, 1] to [0, 255]

    Args:
        images: numpy array of scaled images

    Returns:
        The rescaled images as numpy array
    """
    images = (images * 127.5) + 127.5
    images = np.squeeze(np.round(images).astype(np.uint8))
    return images


def load_img(path, grayscale=False, target_size=None):
    """
    Loads an image into PIL format.

    Args:
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.

    Returns:
        A PIL Image instance.

    Raises:
        ImportError: if PIL is not available.

    References:
        See https://github.com/lim-anggun/Keras-ImageDataGenerator/blob/master/image.py for original source code
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img


class ImageLoader:
    def __init__(self, data: pd.DataFrame, path_column: str, image_size: int = 256, image_ratio: tuple = (1, 1),
                 binarizer: MultiLabelBinarizer = None, color_mode: str = "rgb", row_axis: int = 0, col_axis: int =
                 1, channel_axis: int = 2, rotation_range: int = 0, height_shift_range: float = 0.0,
                 width_shift_range: float = 0.0, shear_range: float = 0.0, zoom_range: tuple = (0.0, 0.0),
                 channel_shift_range: float = 0.0, horizontal_flip: bool = False, vertical_flip: bool = False):
        """
        Image Loader that takes a pandas dataframe containing the file paths to the images and additional meta
        information and loads the existing images. Also allows several image data augmentation applications.

        Args:
            data: pandas dataframe containing the file paths to the images and additional meta information
            path_column: column name that holds the path to the files
            image_size: output size of the images
            image_ratio: output ratio of the images
            binarizer: binarizer used to create dummy features out of additional meta information
            color_mode: `rgb` or `grayscale`
            row_axis: row axis index of the image matrix
            col_axis: column axis index of the image matrix
            channel_axis: channel axis index of the image matrix
            rotation_range: rotation range used for image augmentation
            height_shift_range: height shift range used for image augmentation
            width_shift_range: width shift range used for image augmentation
            shear_range: shear range used for image augmentation
            zoom_range: zoom range used for image augmentation
            channel_shift_range: channel shift range used for image augmentation
            horizontal_flip: whether to flip the images horizontally during image augmentation
            vertical_flip: whether to flip the images horizontally during image augmentation
        """
        self.data = data
        self.path_column = path_column
        self.binarizer = binarizer
        self.image_shape = (np.int(np.ceil(image_size * image_ratio[1])), np.int(np.ceil(image_size * image_ratio[0])))
        self.color_mode = color_mode
        self._iterator = 0
        self._max_erosion = 1
        self.row_axis = row_axis
        self.col_axis = col_axis
        self.channel_axis = channel_axis
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def random_transform(self, x, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            np.random.seed(seed)

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range and np.random.random() > .5:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range and np.random.random() > .5:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range and np.random.random() > .5:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range and np.random.random() > .5:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1 and np.random.random() > .5:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None and np.random.random() > .5:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis, fill_mode="constant", cval=0)

        if self.channel_shift_range != 0 and np.random.random() > .5:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)
        if self.horizontal_flip and np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)

        if self.vertical_flip and np.random.random() < 0.5:
            x = flip_axis(x, img_row_axis)

        return x

    def next(self, batch_size: int = 32, year: bool = False, bool=False, genre: bool = False):
        """
        Loads the next batch of images.

        Args:
            batch_size: size of the drawn batch
            year: whether to load the release year information as well.
            genre: whether to load the binarized genre information as well. Can be nused for genre embedding.

        Returns:
            List if return values. Contains numpy array of images and release year information as well as genre
                information if requested
        """
        batch_x = np.zeros((batch_size, self.image_shape[0], self.image_shape[1], 3), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        year_x = np.zeros((batch_size, 1)) if year else None
        genres_x = np.zeros((batch_size, len(self.binarizer.classes_))) if genre else None

        for i in range(0, batch_size):
            file_path = self.data[self.path_column][self._iterator]
            img = load_img(file_path, grayscale=grayscale, target_size=self.image_shape)
            x = img_to_array(img)
            x = scale_images(x)
            batch_x[i] = x
            if year:
                year_x[i] = self.data["album_release"][self._iterator]
            if genre:
                genres_x[i] = self.binarizer.transform([self.data["artist_genre"][self._iterator]])
            self._iterator += 1
            if self._iterator >= len(self.data):
                self._iterator = 0
                self.data = self.data.sample(frac=1).reset_index(drop=True)

        return self._wrap_output(batch_x, genres_x, year_x)

    def load_all(self, year: bool = False, genre: bool = False):
        """
        Loads all images from the data frame

        Args:
            year: whether to load the release year information as well.
            genre: whether to load the binarized genre information as well. Can be nused for genre embedding.

        Returns:
            List of return values. Contains numpy array of images and release year information as well as genre
                information if requested
        """
        batch_x = np.zeros((len(self.data), self.image_shape[0], self.image_shape[1], 3), dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        year_x = np.zeros((len(self.data), 1)) if year else None
        genres_x = np.zeros((len(self.data), len(self.binarizer.classes_))) if genre else None

        for i in tqdm(range(0, len(self.data))):
            file_path = self.data[self.path_column][self._iterator]
            img = load_img(file_path, grayscale=grayscale, target_size=self.image_shape)
            x = img_to_array(img)
            x = scale_images(x)
            batch_x[i] = x
            if year:
                year_x[i] = self.data["album_release"][self._iterator]
            if genre:
                genres_x[i] = self.binarizer.transform([self.data["artist_genre"][self._iterator]])
            self._iterator += 1

        return self._wrap_output(batch_x, genres_x, year_x)

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
