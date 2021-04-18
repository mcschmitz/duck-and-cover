import os
from pathlib import Path

import numpy as np

from config import config
from loader import DataLoader
from networks import ProGAN
from utils.image_operations import plot_final_gif

image_ratio = config.get("image_ratio")
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14]
LATENT_SIZE = 512
PATH = f"progan-release-year-{LATENT_SIZE}"
TRAIN_STEPS = int(1e6)
N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)
GRADIENT_ACC_STEPS = [1, 1, 1, 1, 1, 1, 1]

warm_start = False
starting_from_block = 0

gradient_penalty_weight = 10.0
lp_path = os.path.join(config.get("learning_progress_path"), PATH)
Path(lp_path).mkdir(parents=True, exist_ok=True)
model_dump_path = os.path.join(lp_path, "model")
Path(model_dump_path).mkdir(parents=True, exist_ok=True)

image_width = max(IMAGE_SIZES) * image_ratio[0]
image_height = max(IMAGE_SIZES) * image_ratio[1]

gan = ProGAN(
    img_width=image_width,
    img_height=image_height,
    latent_size=LATENT_SIZE,
    use_gpu=True,
    gradient_penalty_weight=gradient_penalty_weight,
    n_blocks=N_BLOCKS,
)

gan.set_optimizers(
    generator_optimizer={"lr": 0.001, "betas": (0.0, 0.99)},
    discriminator_optimizer={"lr": 0.001, "betas": (0.0, 0.99)},
)

if warm_start:
    gan.load(path=model_dump_path)

for block in range(starting_from_block, N_BLOCKS):
    image_size = IMAGE_SIZES[block]
    data_loader = DataLoader(
        image_size=image_size,
        batch_size=BATCH_SIZE[block],
        return_release_year=True,
        meta_data_path="data/album_data_frame.json",
    )

    batch_size = BATCH_SIZE[block]
    gan.train(
        data_loader=data_loader,
        block=block,
        global_steps=TRAIN_STEPS,
        batch_size=batch_size,
        path=lp_path,
        write_model_to=model_dump_path,
    )
    gan.save(model_dump_path)

plot_final_gif(path=lp_path)
