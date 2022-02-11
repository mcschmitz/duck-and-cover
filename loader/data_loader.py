from typing import Dict

import numpy as np
import pandas as pd
import torch
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from utils import logger


class DataLoader:
    def __init__(
        self,
        meta_data_path: str,
        image_size: int = 256,
        batch_size: int = 32,
        return_release_year: bool = False,
    ):
        """
        Image loader that takes a path and crawls the directory and
        subdirectories for images and loads them.

        Args:
            meta_data_path: Path to the json file that contains information
                about the training data.
            image_size: output size of the images
            batch_size: Size of one Batch
            return_release_year: Flag to return release year information
        """
        self._iterator_i = 0
        self.image_size = image_size
        self.batch_size = batch_size
        self.meta_df = pd.read_json(
            meta_data_path, orient="records", lines=True
        )
        self.meta_df = self.meta_df.dropna(
            subset=["file_path_64", "file_path_300"]
        )
        self.files = (
            self.meta_df["file_path_64"]
            if self.image_size <= 64
            else self.meta_df["file_path_300"]
        )
        self.files = self.files.to_list()

        self.return_release_year = return_release_year
        if self.return_release_year:
            self.meta_df = self.meta_df.dropna(subset=["album_release"])
            self.release_year_scaler = StandardScaler().fit(
                self.meta_df["album_release"].values.reshape(-1, 1)
            )
        self.n_images = len(self.meta_df)

    def __iter__(self):
        yield from (self[batch_id] for batch_id in range(len(self)))

    def __len__(self):
        return self.n_images // self.batch_size

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        batch_x = np.zeros(
            (self.batch_size, 3, self.image_size, self.image_size)
        )
        year_x = [] if self.return_release_year else None
        batch_idx = self._get_batch_idx()
        for i, b_idx in enumerate(batch_idx):
            file_path = self.files[b_idx]
            try:
                img = imread(file_path)
            except FileNotFoundError as err:
                logger.error(f"Unable to load {file_path}. Error: {err}")
            img = np.moveaxis(img, -1, 0)
            img = resize(img, (3, self.image_size, self.image_size))
            batch_x[i] = img
            if self.return_release_year:
                year = np.array(self.meta_df["album_release"][b_idx]).reshape(
                    -1, 1
                )
                year = self.release_year_scaler.transform(year)
                year_x.append(year.flatten())
        self._iterator_i = batch_idx[-1]
        images = torch.Tensor(batch_x)
        year = torch.Tensor(year_x) if year_x else None
        return {"images": images, "year": year}

    def _get_batch_idx(self):
        positions = np.arange(
            self._iterator_i, self._iterator_i + self.batch_size
        )
        batch_idx = [
            i if i < self.n_images else i - self.n_images for i in positions
        ]
        if 0 in batch_idx:
            logger.info("Data Generator exceeded. Will shuffle input data.")
            self.meta_df = self.meta_df.sample(frac=1).reset_index(drop=True)
            self.files = (
                self.meta_df["file_path_64"]
                if self.image_size <= 64
                else self.meta_df["file_path_300"]
            )
            self.files = self.files.to_list()
        return batch_idx
