from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from skimage.transform import resize

from config import GANTrainConfig


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
        self.data = [i for i in data]
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self._iterator_i = 0

    def __len__(self):
        return len(self.data) // self.batch_size

    def __iter__(self):
        yield from (self[batch_id] for batch_id in range(len(self)))

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        batch_x = np.zeros(
            (self.batch_size, self.channels, self.image_size, self.image_size)
        )
        batch_idx = self._get_batch_idx()
        for i, b_idx in enumerate(batch_idx):
            img = self.data[b_idx]["image"]
            img = np.array(img)
            img = img.reshape(1, img.shape[0], img.shape[1])
            img = resize(img, (1, self.image_size, self.image_size))
            batch_x[i] = img
        self._iterator_i = batch_idx[-1]
        images = torch.Tensor(batch_x)
        return {"images": images}

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
