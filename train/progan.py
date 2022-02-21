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

add_release_year = False

prefix = ["progan"]
if add_release_year:
    prefix += ["release-year"]
run_name = randomname.get_name()

logger = pl.loggers.WandbLogger(
    project="duck-and-cover",
    entity="mcschmitz",
    tags=["ProGAN"] + prefix,
    name=run_name,
)

prefix = "-".join(prefix)
image_ratio = config.get("image_ratio")
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14]
LATENT_SIZE = 512
PATH = f"{prefix}-{LATENT_SIZE}"
TRAIN_STEPS = int(1e6)
N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)

warm_start = False
starting_from_block = 0

lp_path = os.path.join(config.get("learning_progress_path"), PATH, run_name)
Path(lp_path).mkdir(parents=True, exist_ok=True)
model_dump_path = os.path.join(lp_path, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)

image_width = max(IMAGE_SIZES) * image_ratio[0]
image_height = max(IMAGE_SIZES) * image_ratio[1]

pro_gan = ProGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=LATENT_SIZE,
    n_blocks=N_BLOCKS,
    add_year_information=add_release_year,
)
generator = pro_gan.build_generator()
discriminator = pro_gan.build_discriminator()

pro_gan_task = ProGANTask(
    generator=generator, discriminator=discriminator, block=starting_from_block
)
eval_rate = TRAIN_STEPS // 32
callbacks = [
    GenerateImages(
        every_n_train_steps=eval_rate,
        target_size=(256, 256),
        output_dir=lp_path,
    ),
    pl.callbacks.ModelCheckpoint(
        monitor="train/images_shown",
        dirpath=model_dump_path,
        mode="max",
        verbose=True,
        save_last=True,
        every_n_train_steps=eval_rate,
        every_n_epochs=0,
        save_on_train_epoch_end=False,
    ),
]
trainer = pl.Trainer(
    max_steps=TRAIN_STEPS,
    enable_checkpointing=True,
    logger=logger,
    callbacks=callbacks,
    enable_progress_bar=False,
)

if warm_start:
    pro_gan_task.load_from_checkpoint(path=model_dump_path)

for block in range(pro_gan_task.block, N_BLOCKS):
    image_size = IMAGE_SIZES[block]
    data_loader = DataLoader(
        image_size=image_size,
        batch_size=BATCH_SIZE[block],
        return_release_year=add_release_year,
        meta_data_path="data/album_data_frame.json",
    )
    if add_release_year:
        pro_gan_task.release_year_scaler = data_loader.release_year_scaler
    trainer.fit(pro_gan_task, train_dataloader=data_loader)

    batch_size = BATCH_SIZE[block]
    pro_gan_task.train(
        data_loader=data_loader,
        block=block,
        global_steps=TRAIN_STEPS,
        batch_size=batch_size,
        path=lp_path,
        write_model_to=model_dump_path,
    )
    gan.save(model_dump_path)

plot_final_gif(path=lp_path)
