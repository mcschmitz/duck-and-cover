import os
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch_lightning.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from networks.modules.progan import ProGANGenerator
from networks.modules.stylegan import StyleGANGenerator


class GenerateImages(Callback):
    def __init__(
        self,
        every_n_train_steps: int,
        output_dir: str,
        meta_data_path: pd.DataFrame = None,
        target_size: Tuple = (64, 64),
        release_year_scaler: StandardScaler = None,
    ):
        """
        Generates a list of images by predicting with the given generator.

        Feeds normal distributed random numbers into the generator to generate
        images, tiles the image, rescales it and saves the output to a PNG
        file. If the trainer has a W&B logger also logs the images to W&B.

        Args:
            every_n_train_steps: How often the logger should run
            output_dir: Where to save the results
            meta_data_path: Path to the meta data json file. The meta data in
                this file will be used to generate images.
            target_size: Target size of the image in pixels
            release_year_scaler: Standaradscaler that will be used to
                standardize the release year information
        """
        self.every_n_train_steps = every_n_train_steps
        self.target_size = target_size
        self.output_dir = output_dir
        self.data = pd.DataFrame()
        if meta_data_path:
            self.data = pd.read_json(
                meta_data_path, orient="records", lines=True
            )
        self.release_year_scaler = release_year_scaler

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        callbacks,
        batch,
        batch_idx,
    ):
        """
        After a batch has been trained checks if the images should be generated
        at this step.

        Args:
            trainer: PTLightning Trainer
            pl_module: The CoverGANTask
            callbacks: The callbacks list of the trainer
            batch: The current batch
            batch_idx: The current batch idx
        """
        if trainer.global_step % self.every_n_train_steps == 0:
            self.generate_images(pl_module, trainer)

    def on_train_end(self, trainer, pl_module):
        """
        Plots a final set of images after the training.

        Args:
            trainer: PTLightning Trainer
            pl_module: The CoverGANTask
        """
        self.generate_images(pl_module, trainer)

    def generate_images(self, task: pl.LightningModule, trainer: pl.Trainer):
        """
        Generates a list of images by predicting with the given generator.

        Feeds normal distributed random numbers into the generator to
        generate `n_imgs`, tiles the image, rescales it and and saves
        the output to a PNG file.

        Args:
            task: The CoverGANTask
            trainer: PTLightning Trainer
        """
        if self.data.empty:
            n_imgs = 10
        else:
            n_imgs = len(self.data)
        for s in range(n_imgs):
            self.generate_image_set(s, task, trainer)

    def generate_image_set(
        self, s: int, task: pl.LightningModule, trainer: pl.Trainer
    ):
        """
        Generates a set of images after receiving a certain seed.

        Args:
            s: Seed to be used to generate the latent data
            task: The CoverGANTask
            trainer: PTLightning Trainer
        """
        np.random.seed(s)
        task.generator.eval()
        latent_size = task.generator.latent_size
        scaled_year_vec = None
        if self.release_year_scaler is not None:
            if self.data.empty:
                year = [[s]]
            else:
                year = [[(self.data.loc[s, "album_release"])]]
            scaled_year = self.release_year_scaler.transform(year)
            scaled_year_vec = np.repeat(scaled_year, 10).reshape(-1, 1)
            latent_size -= 1
        x0 = np.random.normal(size=latent_size)
        x1 = np.random.normal(size=latent_size)
        x = np.linspace(x0, x1, 10)
        x = Tensor(x)
        x = x.to(task.device)
        fig = self.create_figure(task, x, scaled_year_vec)
        images_shown = trainer.logged_metrics["train/images_shown"]
        images_shown = str(int(images_shown))
        if self.data.empty:
            caption = f"Seed {s}"
        else:
            year = str(self.data.loc[s, "album_release"])
            artist_name = str(self.data.loc[s, "artist_name"])
            album_name = str(self.data.loc[s, "album_name"])
            genre = ", ".join(eval(str(self.data.loc[s, "artist_genre"])))
            caption = f"{artist_name} - {album_name} ({genre}) [{year}]"
        if trainer.logger:
            trainer.logger.log_image(
                key=caption, images=[fig], caption=[caption]
            )
        caption = f"{caption} (step {images_shown})"
        img_path = os.path.join(self.output_dir, f"{caption}.png")
        plt.savefig(img_path)
        plt.close()

    def create_figure(
        self, task: pl.LightningModule, x: Tensor, year: np.array = None
    ) -> plt.Figure:
        """
        Creates a matplotlib figure of generated album covers.

        Args:
            task: The CoverGANTask that contains the generator that should be
                used to generate the images from the latent vectors
            x: Latent vectors
            year: Standardized release year of the artificial album
        """
        if year is not None:
            year = Tensor(year).to(task.device)
        idx = 1
        figsize = (np.array(self.target_size) * [10, 1]).astype(int) / 300
        fig = plt.figure(figsize=figsize, dpi=300)
        with torch.no_grad():
            if isinstance(task.generator, ProGANGenerator):
                output = task.ema_generator(
                    x, year=year, block=task.block, alpha=task.alpha
                )
            elif isinstance(task.generator, StyleGANGenerator):
                output = task.ema_generator(
                    x,
                    year=year,
                    block=task.block,
                    alpha=task.alpha,
                    freeze_noise=True,
                )
            else:
                output = task.generator(x)
        generated_images = output.detach().cpu().numpy()
        generated_images = np.moveaxis(generated_images, 1, -1)
        for img in generated_images:
            if img.shape[-1] == 1:
                img = np.tile(img, (1, 1, 3))
            img = array_to_img(img, scale=True)
            img = img.resize(size=self.target_size)
            plt.subplot(1, 10, idx)
            plt.axis("off")
            plt.imshow(img)
            idx += 1
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1
            )
        return fig


def array_to_img(x: np.ndarray, scale=True) -> Image:
    """
    Converts a 3D Numpy array to a PIL Image instance.

    Args:
        x: Input data, in any form that can be converted to a Numpy array.
            "channels_last". Defaults to `None`, in which case the global
            setting `tf.keras.backend.image_data_format()` is used (unless you
            changed it, it defaults to "channels_last").
        scale: Whether to rescale the image such that minimum and maximum values
            are 0 and 255 respectively. Defaults to `True`.
    """
    if scale:
        x = x - np.min(x)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    return Image.fromarray(x.astype("uint8"), "RGB")
