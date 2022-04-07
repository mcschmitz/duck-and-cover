from pathlib import Path
from typing import Dict, List

import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam

from config import GANTrainConfig
from utils import logger
from utils.callbacks import GenerateImages


class CoverGANTask(pl.LightningModule):
    def __init__(
        self,
        config: GANTrainConfig,
        generator: nn.Module,
        discriminator: nn.Module,
    ):
        """
        Task to train a GAN.

        Args:
            config: Training configuration object.
            generator: PyTorch module that generates images.
            discriminator: PyTorch module that discriminates between real and
                generated images.
        """
        super(CoverGANTask, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.config = config
        self.wandb_run_id = None
        self.images_shown = 0
        self.automatic_optimization = False
        self.wandb_run_name = None

    def configure_optimizers(self):
        """
        Assign both the discriminator and the generator optimizer.
        """
        discriminator_optimizer = Adam(
            params=self.discriminator.parameters(),
            lr=self.config.disc_lr,
            betas=self.config.disc_betas,
        )
        generator_optimizer = Adam(
            self.generator.parameters(),
            lr=self.config.gen_lr,
            betas=self.config.gen_betas,
        )
        return generator_optimizer, discriminator_optimizer

    def configure_callbacks(self) -> List[pl.Callback]:
        """
        Creates the callbacks for the training.

        Will add a ModelCheckpoint Callback to save the model at every n
        steps (n has to be defined in the config that is passed during
        model initialization) and a GenerateImages callback to generate
        images at every n steps.
        """
        return [
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
            ),
            GenerateImages(
                data=self.config.test_data_path,
                add_release_year=self.config.add_release_year,
                every_n_train_steps=self.config.eval_rate,
                output_dir=self.config.learning_progress_path,
            ),
        ]

    def on_fit_start(self):
        """
        This method gets executed before a Trainer trains this model.

        It tells the W&B logger to watch the model in order to check the
        gradients report the gradients if W&B is online.
        """
        Path(self.config.learning_progress_path).mkdir(
            parents=True, exist_ok=True
        )
        if hasattr(self, "logger"):
            if isinstance(self.logger, pl.loggers.WandbLogger):
                try:
                    self.logger.watch(self)
                except ValueError:
                    logger.info("The model is already on the watchlist")
                self.wandb_run_id = self.logger.experiment.id
                self.wandb_run_name = self.logger.experiment.name

    def on_save_checkpoint(self, checkpoint: Dict):
        """
        Adds the W&B run id and run name to the checkpoint.

        Args:
            checkpoint: The checkpoint that is saved.
        """
        checkpoint["wandb_run_id"] = self.wandb_run_id
        checkpoint["wandb_run_name"] = self.wandb_run_name

    def on_load_checkpoint(self, checkpoint: Dict):
        """
        Sets the W&B run id and run name from the loaded checkpoint.

        Args:
            checkpoint: Loaded checkpoint.
        """
        self.wandb_run_id = checkpoint["wandb_run_id"]
        self.wandb_run_name = checkpoint["wandb_run_name"]
