import os
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import tensorflow as tf
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback


class GenerateImages(Callback):
    def __init__(
        self,
        every_n_train_steps: int,
        output_dir: str,
        target_size: Tuple = (64, 64),
        **kwargs,
    ):
        """
        Generates a list of images by predicting with the given generator.

        Feeds normal distributed random numbers into the generator to generate
        `n_imgs`, tiles the image, rescales it and saves the output to a PNG
        file. If the trainer has a W&B logger also logs the images to W&B.

        Args:
            every_n_train_steps: How often the logger should run
            output_dir: Where to save the results
            n_imgs: Number of images to generate
            target_size: Target size of the image in pixels
            **kwargs: Additional keyword arguments
        """
        self.every_n_train_steps = every_n_train_steps
        self.target_size = target_size
        self.release_year_scaler = kwargs.get("release_year_scaler", None)
        self.output_dir = output_dir

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
        for s in range(25):
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
        if self.release_year_scaler:
            year = np.random.randint(1925, 2025, 1).reshape(-1, 1)
            scaled_year = self.release_year_scaler.transform(year)
            scaled_year_vec = np.repeat(scaled_year, 10).reshape(-1, 1)
            latent_size -= 1
        x0 = np.random.normal(size=latent_size)
        x1 = np.random.normal(size=latent_size)
        x = np.linspace(x0, x1, 10)
        if self.release_year_scaler:
            x = np.hstack([x, scaled_year_vec])
        x = torch.Tensor(x)
        x = x.to(task.device)
        fig = self.create_figure(task, x)
        if self.release_year_scaler:
            fig.axes[0].text(
                2,
                15,
                f"Release year: {year[0][0]}",
                color="black",
                fontsize=4,
            )
        images_shown = trainer.logged_metrics["train/images_shown"]
        images_shown = str(int(images_shown))
        if trainer.logger:
            caption = f"seed_{s}"
            trainer.logger.log_image(key=caption, images=[fig])
        caption = f"{s}_step_{images_shown}"
        img_path = os.path.join(self.output_dir, f"{caption}.png")
        plt.savefig(img_path)
        plt.close()

    def create_figure(
        self, task: pl.LightningModule, x: torch.Tensor
    ) -> plt.Figure:
        """
        Creates a matplotlib figure of generated album covers.

        Args:
            task: The CoverGANTask that contains the generator that should be
                used to generate the images from the latent vectors
            x: Latent vectors
        """
        idx = 1
        figsize = (np.array(self.target_size) * [10, 1]).astype(int) / 300
        fig = plt.figure(figsize=figsize, dpi=300)
        output = task.generator(x, block=task.block, alpha=task.alpha)
        generated_images = output.detach().cpu().numpy()
        generated_images = np.moveaxis(generated_images, 1, -1)
        for img in generated_images:
            img = tf.keras.utils.array_to_img(img, scale=True)
            img = img.resize(size=self.target_size)
            plt.subplot(1, 10, idx)
            plt.axis("off")
            plt.imshow(img)
            idx += 1
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.1
            )
        return fig
