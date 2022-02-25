from typing import Dict

import numpy as np
import torch
from torch import nn

from tasks.covergantask import CoverGANTask


class WGANTask(CoverGANTask):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        name: str,
        n_critic: int = 5,
        gradient_penalty_weight: float = 10.0,
    ):
        """
        Wasserstein GAN training task. Takes an generator and discriminator and
        can be used to train a Wasserstein GAN with PyTorch Lightning.

        Args:
            generator: PyTorch Generator Model
            discriminator: PyTorch Discriminator Model
            n_critic: Number of times the discriminator should see the data
                before the generator gets trained.
            gradient_penalty_weight: Weight of the gradient penalty for the
                W-GAN
        """
        super(WGANTask, self).__init__(
            generator=generator, discriminator=discriminator, name=name
        )
        self.n_critic = n_critic
        self.gradient_penalty_weight = gradient_penalty_weight

    def train_on_batch(self, batch: Dict[str, torch.Tensor]):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: The batch that should be used for training. Should be a Dict
                mapping strings to tensors
        """
        d_accuracies = []
        for _ in range(self.n_critic):
            noise = torch.normal(
                mean=0,
                std=1,
                size=(len(batch["images"]), self.discriminator.latent_size),
            )
            d_accuracies.append(self.train_discriminator(batch, noise))
        d_accuracies = np.mean([d.detach().tolist() for d in d_accuracies])
        self.log("train/discriminator_loss", np.mean(d_accuracies))

        noise = torch.normal(
            mean=0,
            std=1,
            size=(len(batch["images"]), self.discriminator.latent_size),
        )
        self.log("train/generator_loss", self.train_generator(batch, noise))

    def _gradient_penalty(
        self,
        real_batch: Dict[str, torch.Tensor],
        fake_batch: Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        batch_size = real_batch["images"].shape[0]

        random_avg_weights = torch.rand((batch_size, 1, 1, 1)).to(
            real_batch["images"].device
        )
        random_avg = random_avg_weights * real_batch["images"] + (
            (1 - random_avg_weights) * fake_batch["images"]
        )
        random_avg.requires_grad_(True)

        pred = self.discriminator(
            images=random_avg, year=real_batch["year"], **kwargs
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
        return (
            self.gradient_penalty_weight
            * ((grad.norm(p=2, dim=1) - 1) ** 2).mean()
        )
