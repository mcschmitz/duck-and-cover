from typing import Dict

import torch
from torch import Tensor, nn

from config import GANTrainConfig
from tasks.progan import ProGANTask


class StyleGANTask(ProGANTask):
    def __init__(
        self,
        config: GANTrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        super().__init__(
            config=config,
            generator=generator,
            discriminator=discriminator,
        )
        self.block += 1
        self.softplus = nn.Softplus()
        self.current_resolution = 2 ** (self.block + 2)

    def on_fit_start(self):
        super().on_fit_start()
        self.current_resolution = 2 ** (self.block + 2)

    def train_on_batch(self, batch: Dict[str, Tensor], batch_idx: int):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Batch of data to train on
            batch_idx: Incremental batch index
        """
        self.train_discriminator(batch)
        if batch_idx % self.config.n_critic == 0:
            self.train_generator(batch=batch)

    def train_generator(self, batch: Dict[str, torch.Tensor]):
        """
        Trains the generator on a single batch of data and returns the
        Generator loss.

        Args:
            batch: Input batch containing images and year
        """
        generator_optimizer, _ = self.optimizers()
        generator_optimizer.zero_grad()

        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        noise = noise.to(self.device)
        generated_images = self.generator(
            x=noise, year=batch.get("year"), block=self.block, alpha=self.alpha
        )
        scores_generated_images = self.discriminator(
            images=generated_images,
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        g_loss = torch.mean(self.softplus(-scores_generated_images))
        self.manual_backward(g_loss)
        generator_optimizer.step()
        self.trainer.logger.log_metrics(
            {"train/generator_loss": g_loss}, step=self.images_shown
        )
        self.update_ema_generator(self.config.ema_beta)

    def train_discriminator(self, batch: Dict[str, torch.Tensor]):
        """
        Runs a single gradient update on a batch of data.

        Args:
            batch: Real input images used for training
        """
        _, discriminator_optimizer = self.optimizers()
        discriminator_optimizer.zero_grad()

        noise = torch.normal(
            mean=0, std=1, size=(len(batch["images"]), self.config.latent_size)
        )
        noise = noise.to(self.device)
        with torch.no_grad():
            generated_images = self.generator(
                x=noise,
                year=batch.get("year"),
                block=self.block,
                alpha=self.alpha,
            ).detach()

        real_imgs = batch["images"]
        real_logits = self.discriminator(
            images=real_imgs,
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        fake_logits = self.discriminator(
            images=generated_images,
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        scores_fake = torch.mean(self.softplus(fake_logits))
        scores_real = torch.mean(self.softplus(-real_logits))
        loss = scores_fake + scores_real
        real_imgs = torch.autograd.Variable(real_imgs, requires_grad=True)
        real_logit = self.discriminator(
            images=real_imgs,
            year=batch.get("year"),
            block=self.block,
            alpha=self.alpha,
        )
        real_grads = torch.autograd.grad(
            outputs=real_logit,
            inputs=real_imgs,
            grad_outputs=torch.ones(real_logit.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
        )[0].view(real_imgs.size(0), -1)
        r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
        loss += r1_penalty
        self.manual_backward(loss)
        discriminator_optimizer.step()

        batch_size = self.config.batch_size[self.current_resolution]
        real = torch.zeros((batch_size, 1)).to(self.device)
        fake = torch.ones((batch_size, 1)).to(self.device)
        fake_pred = self.sigmoid(fake_logits)
        real_pred = self.sigmoid(real_logits)
        correct_fake = torch.sum(torch.round(fake_pred) == fake).cpu().numpy()
        correct_real = torch.sum(torch.round(real_pred) == real).cpu().numpy()
        discriminator_acc_real = correct_real / batch_size
        discriminator_acc_fake = correct_fake / batch_size
        discriminator_acc = (correct_real + correct_fake) / (batch_size * 2)
        self.logger.log_metrics(
            {
                "train/discriminator_loss": loss.cpu().item(),
                "train/discriminator_acc": discriminator_acc,
                "train/r1_penalty": r1_penalty.cpu().item(),
                "train/discriminator_acc_real": discriminator_acc_real,
                "train/discriminator_acc_fake": discriminator_acc_fake,
                "train/scores_real": real_logits.mean().cpu().item(),
                "train/scores_fake": fake_logits.mean().cpu().item(),
            },
            step=self.images_shown,
        )
