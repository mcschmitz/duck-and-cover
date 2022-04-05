from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from skimage.io import imread
from skimage.transform import resize

from config import GANTrainConfig
from utils import logger


class MNISTTrainGenerator:
    def __init__(self, data, batch_size: int, image_size: int, channels: int):
        """
        Training dataloader for the MNIST dataset.

        Args:
            data: MNIST dataset loaded from huggingface datasets.
            batch_size: Training batch size.
            image_size: Image size.
            channels: Number of channels.
        """
        self.data = data
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        yield from (self[batch_id] for batch_id in range(len(self)))

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        batch_x = np.zeros(
            (self.batch_size, self.channels, self.image_size, self.image_size)
        )
        year_x = [] if self.add_release_year else None
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
            if self.add_release_year:
                year = np.array(self.meta_df["album_release"][b_idx]).reshape(
                    -1, 1
                )
                year = self.release_year_scaler.transform(year)
                year_x.append(year.flatten())
        self._iterator_i = batch_idx[-1]
        images = torch.Tensor(batch_x)
        year = torch.Tensor(np.array(year_x)) if year_x else None
        return {"images": images, "year": year}


class MNISTDataloader:
    def __init__(self, config: GANTrainConfig):
        """
        Dataloader for the MNIST dataset.

        Args:
            config: Training configuration object.
        """
        self.config = config
        self.dataset = load_dataset("mnist")

    def __len__(self):
        return len(self.dataset["train"]) // self.config.batch_size

    def get_data_generators(self) -> Dict:
        """
        Returns the training and validation data generator.
        """
        return {
            "train": MNISTTrainGenerator(
                data=self.dataset["train"],
                batch_size=self.config.batch_size,
                image_size=self.config.image_size,
                channels=1,
            )
        }
