import os
from pathlib import Path

import numpy as np

from config import config
from loader import DataLoader
from networks import ProGAN
from utils import logger
from utils.image_operations import plot_final_gif

image_ratio = config.get("image_ratio")
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14]
LATENT_SIZE = 512
PATH = f"progan-{LATENT_SIZE}"
TRAIN_STEPS = int(1e6)
N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)
GRADIENT_ACC_STEPS = [1, 1, 1, 1, 1, 1, 1]

warm_start = False
starting_from_block = 1

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

for block in range(starting_from_block, N_BLOCKS):
    image_size = IMAGE_SIZES[block]
    data_path = os.path.join(
        config.get("base_data_path"), f"covers{300 if image_size > 64 else 64}"
    )
    data_loader = DataLoader(
        data_path, image_size=image_size, batch_size=BATCH_SIZE[block]
    )
    if warm_start:
        model_path = os.path.join(lp_path, f"model-{image_size}x{image_size}")
        model_path_pre = os.path.join(
            lp_path, f"model-{image_size//2}" f"x{image_size//2}"
        )
        if os.path.isfile(os.path.join(model_path, "C_fade_in.h5")):
            logger.info(f"Load model from {model_path}")
            for _ in range(starting_from_block):
                gan.add_block(
                    optimizer=optimizer,
                    gradient_accumulation_steps=GRADIENT_ACC_STEPS[block],
                )
            gan.load(model_path)
        elif os.path.isfile(os.path.join(model_path_pre, "C.h5")):
            for _ in range(starting_from_block - 1):
                gan.add_block(
                    optimizer=optimizer,
                    gradient_accumulation_steps=GRADIENT_ACC_STEPS[block],
                )
            logger.info(f"Load model from {model_path_pre}")
            gan.load(model_path_pre)
            gan.add_block(
                optimizer=optimizer,
                gradient_accumulation_steps=GRADIENT_ACC_STEPS[block],
            )
        else:
            raise ValueError("Model not found")
        warm_start = False

    model_dump_path = os.path.join(lp_path, f"model-{image_size}x{image_size}")
    Path(model_dump_path).mkdir(parents=True, exist_ok=True)

    batch_size = BATCH_SIZE[block]
    gan.train(
        data_loader=data_loader,
        block=block,
        global_steps=TRAIN_STEPS,
        batch_size=batch_size,
        verbose=True,
        path=lp_path,
        write_model_to=model_dump_path,
    )
    gan.save(model_dump_path)
    gan.add_block(
        optimizer=optimizer,
        gradient_accumulation_steps=GRADIENT_ACC_STEPS[block],
    )

plot_final_gif(path=lp_path)
