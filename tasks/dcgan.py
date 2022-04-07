import torch
from torch import Tensor, nn

from config import GANTrainConfig
from tasks.covergantask import CoverGANTask


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
            generator: PyTorch module that generates images.
            discriminator: PyTorch module that discriminates between real and
                generated images.
        """
        super().__init__(
            config=config, generator=generator, discriminator=discriminator
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def training_step(self, batch):
        """
        Trains the GAN on the given batch.

        Args:
            batch: Batch to train on
        """
        self.train_on_batch(batch)
        self.images_shown += len(batch["images"])
        self.log("train/images_shown", self.images_shown)

    def train_on_batch(self, batch: Tensor, **kwargs):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Batch of data to train on
            kwargs: Additional arguments to pass to the training functions
        """
        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        generated_images = self.train_generator(noise)
        self.train_discriminator(batch, generated_images)

    def train_discriminator(self, batch, generated_images):
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()
        real = torch.zeros((self.config.batch_size, 1)).to(self.device)
        fake = torch.ones((self.config.batch_size, 1)).to(self.device)

        real_pred = self.discriminator(batch["images"])
        fake_pred = self.discriminator(generated_images.detach())
        loss_fake = self.loss(fake_pred, fake)
        loss_real = self.loss(real_pred, real)
        fake_pred = self.sigmoid(fake_pred)
        real_pred = self.sigmoid(real_pred)
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        discriminator_acc = (correct_real + correct_fake) / (
            self.config.batch_size * 2
        )
        d_loss = (loss_fake + loss_real) / 2
        if discriminator_acc < 0.5:
            self.manual_backward(d_loss)
            discriminator_optimizer.step()
        self.log("train/discriminator_loss", d_loss)
        self.log("train/discriminator_accuracy", discriminator_acc)

    def train_generator(self, noise):
        real = torch.zeros((self.config.batch_size, 1)).to(self.device)
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()
        noise = noise.to(self.device)
        generated_images = self.generator(noise)
        g_loss = self.loss(self.discriminator(generated_images), real)
        self.manual_backward(g_loss)
        generator_optimizer.step()
        self.log("train/generator_loss", g_loss)
        return generated_images
