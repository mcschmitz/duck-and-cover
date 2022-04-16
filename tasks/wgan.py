from typing import Dict

import torch
from torch import Tensor, nn

from config import GANTrainConfig
from tasks.dcgan import DCGanTask


class WGANTask(DCGanTask):
    def __init__(
        self,
        config: GANTrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Wasserstein GAN training task. Takes an generator and discriminator and
        can be used to train a Wasserstein GAN with PyTorch Lightning.

        Args:
            config: Config object containing the model hyperparameters
            generator: PyTorch module that generates images.
            discriminator: PyTorch module that discriminates between real and
                generated images.
        """
        super().__init__(
            config=config, generator=generator, discriminator=discriminator
        )

    def train_on_batch(self, batch: Dict[str, Tensor], batch_idx: int):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Batch of data to train on
            batch_idx: Incremental batch index
        """
        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        noise = noise.to(self.device)
        generated_images = self.generator(noise)
        self.train_discriminator(batch, generated_images)
        if batch_idx % self.config.n_critic == 0:
            self.train_generator(noise)

    def train_discriminator(
        self, batch: Dict[str, Tensor], generated_images: Tensor
    ):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Batch of data to train on
            generated_images: Images generated by the generator
        """
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()

        real_pred = self.discriminator(batch["images"])
        fake_pred = self.discriminator(generated_images)

        loss = torch.mean(fake_pred) - torch.mean(real_pred)
        gp = self._gradient_penalty(batch, generated_images)
        loss += self.config.gradient_penalty_weight * gp
        self.manual_backward(loss)
        discriminator_optimizer.step()

        real = torch.zeros((self.config.batch_size, 1)).to(self.device)
        fake = torch.ones((self.config.batch_size, 1)).to(self.device)
        fake_pred = self.sigmoid(fake_pred)
        real_pred = self.sigmoid(real_pred)
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        discriminator_acc = (correct_real + correct_fake) / (
            self.config.batch_size * 2
        )
        self.log("train/discriminator_loss", loss)
        self.log("train/discriminator_accuracy", discriminator_acc)

    def train_generator(self, noise: Tensor):
        """
        Trains the generator.

        Args:
            noise: Tenor of noise to generate images from
        """
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()
        noise = noise.to(self.device)
        generated_images = self.generator(noise)
        g_loss = torch.mean(-self.discriminator(generated_images))
        self.manual_backward(g_loss)
        generator_optimizer.step()
        self.log("train/generator_loss", g_loss)

    def _gradient_penalty(
        self,
        real_batch: Dict[str, Tensor],
        fake_batch: Tensor,
        **kwargs,
    ) -> Tensor:
        batch_size = real_batch["images"].shape[0]

        random_avg_weights = torch.rand((batch_size, 1, 1, 1)).to(self.device)
        random_avg = random_avg_weights * real_batch["images"] + (
            (1 - random_avg_weights) * fake_batch
        )
        random_avg.requires_grad_(True)
        random_avg = random_avg.to(self.device)
        pred = self.discriminator(
            images=random_avg,
            year=real_batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        grad = torch.autograd.grad(
            outputs=pred,
            inputs=random_avg,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grad.view(grad.shape[0], -1)
        return ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
