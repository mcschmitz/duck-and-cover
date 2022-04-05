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
        """
        Trains the ProGAN on a single batch of data. Obtains the phase (`burn-
        in` or `phade-in`) and trains the GAN accordingly.

        Args:
            batch: Batch to train on
        """
        self.train_on_batch(batch)
        self.log(
            "train/images_shown",
            len(batch["images"]),
        )

    def train_on_batch(self, real_images: Tensor, **kwargs):
        """
        Runs a single gradient update on a batch of data.

        Args:
            real_images: numpy array of real input images used for training
        """
        noise = torch.normal(
            mean=0, std=1, size=(len(real_images), self.latent_size)
        )
        self.metrics["D_accuracy"]["values"].append(
            self.train_discriminator(real_images, noise)
        )
        self.metrics["G_loss"]["values"].append(self.train_generator(noise))

    def train_discriminator(
        self, real_images: torch.Tensor, noise: torch.Tensor
    ) -> float:
        """
        Runs a single gradient update for the discriminator.

        Args:
            real_images: Array of real images
            noise: Random noise input Tensor
        """
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()
        with torch.no_grad():
            generated_images = self.generator(noise)
        fake = torch.ones((len(generated_images), 1))
        real = torch.zeros((len(real_images), 1))
        if self.use_gpu:
            fake = fake.cuda()
            real = real.cuda()
        fake_pred = self.discriminator(generated_images)
        real_pred = self.discriminator(real_images)
        loss_fake = nn.BCELoss()(fake_pred, fake)
        loss_real = nn.BCELoss()(real_pred, real)
        loss = (loss_fake + loss_real) / 2
        loss.backward()
        self.discriminator_optimizer.step()
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        return (correct_real + correct_fake) / (len(generated_images) * 2)

    def train_generator(self, noise: torch.Tensor) -> float:
        """
        Runs a single gradient update for the generator.

        Args:
            noise: Random noise input Tensor
        """
        self.generator.train()
        self.generator_optimizer.zero_grad()
        fake = torch.ones((len(noise), 1))
        if self.use_gpu:
            fake = fake.cuda()
        generated_images = self.generator(noise)
        fake_pred = self.discriminator(generated_images)
        loss_fake = nn.BCELoss()(fake_pred, fake)
        loss_fake.backward()
        self.generator_optimizer.step()
        return loss_fake.detach().cpu().numpy().tolist()
