from typing import List

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from config import GANTrainConfig
from tasks.covergantask import CoverGANTask
from utils.callbacks import GenerateImages


class DCGanTask(CoverGANTask):
    def __init__(
        self,
        config: GANTrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Simple Deep Convolutional GAN.

        Simple DCGAN that builds a discriminator a generator and the adversarial
        model to train the GAN based on the binary crossentropy loss for the
        generator and the discriminator.

        Args:
            config: Config object containing the model hyperparameters
        """
        super().__init__(
            config=config, generator=generator, discriminator=discriminator
        )
        self.loss = nn.BCELoss()

    def configure_callbacks(self) -> List[pl.Callback]:
        cbs = [
            pl.callbacks.ModelCheckpoint(
                monitor="train/images_shown",
                dirpath=self.config.learning_progress_path,
                filename="model_ckpt",
                mode="max",
                verbose=True,
                save_last=True,
                every_n_train_steps=self.config.eval_rate,
                every_n_epochs=0,
                save_on_train_epoch_end=False,
            )
        ]
        if self.config.dataloader == "DataLoader":
            cbs.append(
                GenerateImages(
                    data=test_data_meta,
                    add_release_year=ADD_RELEASE_YEAR,
                    release_year_scaler=data_loader.release_year_scaler,
                )
            )
        return cbs

    def training_step(self, batch):
        self.train_on_batch(batch)
        self.images_shown += len(batch["images"])
        self.log("train/images_shown", self.images_shown)

    def train_on_batch(self, batch: Tensor, **kwargs):
        """
        Runs a single gradient update on a batch of data.
        """
        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        noise = noise.to(self.device)
        acc = self.train_discriminator(batch, noise)
        self.log("train/discriminator_accuracy", acc)
        loss = self.train_generator(noise)
        self.log("train_generator/loss", loss)

    def train_discriminator(
        self, batch: torch.Tensor, noise: torch.Tensor
    ) -> float:
        """
        Runs a single gradient update for the discriminator.

        Args:
            noise: Random noise input Tensor
        """
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()
        with torch.no_grad():
            fake_images = self.generator(noise)
        fake = torch.ones((len(fake_images), 1)).to(self.device)
        real = torch.zeros((len(batch["images"]), 1)).to(self.device)
        fake_pred = self.discriminator(fake_images)
        real_pred = self.discriminator(batch["images"])
        loss_fake = self.loss(fake_pred, fake)
        loss_real = self.loss(real_pred, real)
        loss = (loss_fake + loss_real) / 2
        self.manual_backward(loss)
        discriminator_optimizer.step()
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        return (correct_real + correct_fake) / (len(fake_images) * 2)

    def train_generator(self, noise: torch.Tensor) -> float:
        """
        Runs a single gradient update for the generator.

        Args:
            noise: Random noise input Tensor
        """
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()
        generated_images = self.generator(noise)
        fake = torch.ones((len(noise), 1)).to(self.device)
        fake_pred = self.discriminator(generated_images)
        loss_fake = self.loss(fake_pred, fake)
        self.manual_backward(loss_fake)
        generator_optimizer.step()
        return loss_fake.detach().cpu().numpy().tolist()
