from typing import Dict

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from config import GANTrainConfig


class HFDataloader:
    def __init__(self, config: GANTrainConfig):
        """
        Dataloader for a Huggingface dataset.

        Args:
            config: GANTrainConfig containing the necessary information to
                download the data and set up the datagenerators.
                GANTrainConfig is a pydantic model that needs `dataset_name`,
                `image_size` and `batch_size` to be set.
        """
        self.config = config
        self.dataset = load_dataset(config.dataset_name)
        self.dataset.set_transform(self._transforms)

    def _transforms(self, examples) -> Dict:
        augmentations = Compose(
            [
                Resize(
                    self.config.image_size,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                CenterCrop(self.config.image_size),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )
        images = [
            augmentations(image.convert("RGB")) for image in examples["image"]
        ]
        return {"input": images}

    def train_dataloader(self) -> DataLoader:
        """
        Returns the dataloader.
        """
        return DataLoader(
            self.dataset["train"],
            batch_size=self.config.batch_size,
            shuffle=True,
        )
