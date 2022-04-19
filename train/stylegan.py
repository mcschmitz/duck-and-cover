import argparse
import os

import pytorch_lightning as pl

from config import GANTrainConfig
from networks import StyleGAN
from tasks.stylegan import StyleGANTask
from utils.image_operations import create_final_gif

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    config = GANTrainConfig(args.config_file)
    dataloader = config.get_dataloader()

    style_gan = StyleGAN(config)
    generator = style_gan.build_generator()
    discriminator = style_gan.build_discriminator()

    style_gan_task = StyleGANTask(
        config=config, generator=generator, discriminator=discriminator
    )

    if config.warm_start:
        style_gan_task = style_gan_task.load_from_checkpoint(
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
        tags=["StyleGAN"] + config.wandb_tags,
        name=config.unique_experiment_name,
        id=style_gan_task.wandb_run_id,
    )
    while style_gan_task.block < config.n_blocks:
        block = style_gan_task.block
        image_size = 2 ** (block + 2)
        data_generators = dataloader.get_data_generators(image_size=image_size)
        trainer = pl.Trainer(
            gpus=-1,
            max_steps=config.train_steps - (style_gan_task.phase_steps * 2),
            enable_checkpointing=True,
            logger=logger,
            precision=config.precision,
            enable_progress_bar=False,
        )
        trainer.fit(
            style_gan_task,
            train_dataloaders=data_generators.get("train"),
            val_dataloaders=data_generators.get("val"),
        )
    create_final_gif(path=config.learning_progress_path)
