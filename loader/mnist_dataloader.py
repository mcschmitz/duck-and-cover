from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

from config import GANTrainConfig


class MNISTTrainGenerator:
    def __init__(
        self,
        data,
        batch_size: int,
        image_size: int,
        channels: int,
        return_release_year: bool = False,
    ):
        """
        Training dataloader for the MNIST dataset.

        Args:
            data: MNIST dataset loaded from huggingface datasets.
            batch_size: Training batch size.
            image_size: Image size.
            channels: Number of channels.
            return_release_year: Whether to return the release year.
        """
        self.data = list(data)
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels

        self.return_release_year = return_release_year
        if self.return_release_year:
            unique_years = {sample["label"] for sample in self.data}
            self.release_year_scaler = StandardScaler().fit(
                np.array(list(unique_years)).reshape(-1, 1)
            )

        self._iterator_i = 0

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        yield from (self[batch_id] for batch_id in range(len(self)))

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        batch_x = np.zeros(
            (self.batch_size, self.channels, self.image_size, self.image_size)
        )
        year_x = []

        batch_idx = self._get_batch_idx()
        for i, b_idx in enumerate(batch_idx):
            sample = self.data[b_idx]
            img = sample["image"]
            img = np.array(img)
            img = img.reshape(1, img.shape[0], img.shape[1])
            img = resize(img, (1, self.image_size, self.image_size))
            batch_x[i] = img
            if self.return_release_year:
                year = np.array(sample["label"]).reshape(-1, 1)
                year = self.release_year_scaler.transform(year)
                year_x.append(year.flatten())

        batch_x = (batch_x + 1) / 2
        self._iterator_i = batch_idx[-1]

        images = torch.Tensor(batch_x)
        year = torch.Tensor(np.array(year_x)) if year_x else None
        return {"images": images, "year": year}

    def _get_batch_idx(self):
        positions = np.arange(
            self._iterator_i, self._iterator_i + self.batch_size
        )
        return [
            i if i < len(self.data) else i - len(self.data) for i in positions
        ]


class MNISTDataloader:
    def __init__(self, config: GANTrainConfig):
        """
        Dataloader for the MNIST dataset.

        Args:
            config: Training configuration object.
        """
        self.config = config
        self.dataset = load_dataset("mnist")

    def get_data_generators(
        self, image_size: int = None, return_release_year: bool = False
    ) -> Dict:
        """
        Returns the MNIST training generator.

        Args:
            image_size: Size of the images to be returned by the generator.
            return_release_year: Whether to return the release year.
        """
        image_size = image_size or self.config.image_size
        return {
            "train": MNISTTrainGenerator(
                data=self.dataset["train"],
                batch_size=self.config.batch_size,
                image_size=image_size,
                channels=1,
                return_release_year=return_release_year,
            )
        }
