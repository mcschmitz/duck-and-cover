import argparse
import os

import pytorch_lightning as pl

from config import ProGANTrainConfig
from networks import ProGAN
from tasks.progan import ProGANTask
from utils.image_operations import create_final_gif

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    config = ProGANTrainConfig(args.config_file)
    dataloader = config.get_dataloader()

    pro_gan = ProGAN(config)
    generator = pro_gan.build_generator()
    discriminator = pro_gan.build_discriminator()

    pro_gan_task = ProGANTask(
        config=config, generator=generator, discriminator=discriminator
    )

    if config.warm_start:
        pro_gan_task = pro_gan_task.load_from_checkpoint(
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
        tags=["ProGAN"] + config.wandb_tags,
        name=config.unique_experiment_name,
        id=pro_gan_task.wandb_run_id,
    )
    while pro_gan_task.block < config.n_blocks:
        block = pro_gan_task.block
        resolution = 2 ** (block + 2)
        data_generators = dataloader.get_data_generators(image_size=resolution)
        train_imgs = config.fade_in_imgs if pro_gan_task.phase == "fade_in" else config.burn_in_imgs
        train_steps = train_imgs // data_generators["train"].batch_size
        trainer = pl.Trainer(
            gpus=-1,
            max_steps=train_steps - (pro_gan_task.phase_steps * 2),
            enable_checkpointing=True,
            logger=logger,
            precision=config.precision,
            enable_progress_bar=False,
        )
        trainer.fit(
            pro_gan_task,
            train_dataloaders=data_generators.get("train"),
            val_dataloaders=data_generators.get("val"),
        )
    create_final_gif(path=config.learning_progress_path)
