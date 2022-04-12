import argparse
import os

import pytorch_lightning as pl

from config import GANTrainConfig
from networks import WGAN
from tasks import WGANTask
from utils.image_operations import create_final_gif

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    config = GANTrainConfig(args.config_file)
    dataloader = config.get_dataloader()
    data_generators = dataloader.get_data_generators()

    wgan = WGAN(config)
    generator = wgan.build_generator()
    discriminator = wgan.build_discriminator()

    wgan_task = WGANTask(
        config=config, generator=generator, discriminator=discriminator
    )

    if config.warm_start:
        wgan_task = wgan_task.load_from_checkpoint(
            config=config,
            generator=generator,
            discriminator=discriminator,
            checkpoint_path=os.path.join(
                config.learning_progress_path, "last.ckpt"
            ),
        )

    logger = pl.loggers.WandbLogger(
        project="duck-and-cover",
        entity="mcschmitz",
        tags=["WGAN"] + config.wandb_tags,
        name=config.unique_experiment_name,
        id=wgan_task.wandb_run_id,
    )
    trainer = pl.Trainer(
        gpus=-1,
        max_steps=config.train_steps,
        enable_checkpointing=True,
        logger=logger,
        enable_progress_bar=False,
        precision=config.precision,
    )
    trainer.fit(
        wgan_task,
        train_dataloaders=data_generators.get("train"),
        val_dataloaders=data_generators.get("val"),
    )
    create_final_gif(path=config.learning_progress_path)
