import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import randomname

from config import config
from loader import DataLoader
from networks import ProGAN
from tasks.progan import ProGANTask
from utils import GenerateImages
from utils.image_operations import plot_final_gif

# Model Configs
ADD_RELEASE_YEAR = False
LATENT_SIZE = 512
IMAGE_RATIO = config.get("image_ratio")
N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)
image_width = max(IMAGE_SIZES) * IMAGE_RATIO[0]
image_height = max(IMAGE_SIZES) * IMAGE_RATIO[1]

# Training Configs
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14]
IMAGES_TO_SHOW_PER_PHASE = 8 * int(1e5)
steps_per_batch_size = [IMAGES_TO_SHOW_PER_PHASE // bs for bs in BATCH_SIZE]
warm_start = False

# Experiment Config
prefix_list = ["progan"]
if ADD_RELEASE_YEAR:
    prefix_list += ["release-year"]
run_name = randomname.get_name()
prefix = "-".join(prefix_list)
experiment_path = f"{prefix}-{LATENT_SIZE}"
lp_path = os.path.join(
    config.get("learning_progress_path"), experiment_path, run_name
)
Path(lp_path).mkdir(parents=True, exist_ok=True)
model_dump_path = os.path.join(lp_path, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    pro_gan = ProGAN(
        img_width=image_width,
        img_height=image_height,
        latent_size=LATENT_SIZE,
        n_blocks=N_BLOCKS,
        add_year_information=ADD_RELEASE_YEAR,
    )
    generator = pro_gan.build_generator()
    discriminator = pro_gan.build_discriminator()

    pro_gan_task = ProGANTask(
        generator=generator, discriminator=discriminator, name=run_name
    )
    eval_rate = steps_per_batch_size[0] // 32
    callbacks = [
        GenerateImages(
            every_n_train_steps=eval_rate,
            target_size=(256, 256),
            output_dir=lp_path,
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="train/images_shown",
            dirpath=model_dump_path,
            filename="model_ckpt",
            mode="max",
            verbose=True,
            save_last=True,
            every_n_train_steps=eval_rate,
            every_n_epochs=0,
            save_on_train_epoch_end=False,
        ),
    ]

    if warm_start:
        pro_gan_task = pro_gan_task.load_from_checkpoint(
            generator=generator,
            discriminator=discriminator,
            name=run_name,
            checkpoint_path=os.path.join(model_dump_path, "last.ckpt"),
        )
    logger = pl.loggers.WandbLogger(
        project="duck-and-cover",
        entity="mcschmitz",
        tags=["ProGAN"] + prefix_list,
        name=pro_gan_task.wandb_run_name,
        id=pro_gan_task.wandb_run_id,
    )
    while pro_gan_task.block <= N_BLOCKS - 1:
        block = pro_gan_task.block
        image_size = IMAGE_SIZES[block]
        data_loader = DataLoader(
            image_size=image_size,
            batch_size=BATCH_SIZE[block],
            return_release_year=ADD_RELEASE_YEAR,
            meta_data_path="data/album_data_frame.json",
        )
        if ADD_RELEASE_YEAR:
            pro_gan_task.release_year_scaler = data_loader.release_year_scaler

        max_steps = steps_per_batch_size[block] - pro_gan_task.phase_steps
        trainer = pl.Trainer(
            gpus=-1,
            max_steps=max_steps,
            enable_checkpointing=True,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=False,
        )
        trainer.fit(pro_gan_task, train_dataloader=data_loader)
    plot_final_gif(path=lp_path)
