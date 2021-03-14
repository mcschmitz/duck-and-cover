import os
from pathlib import Path

import numpy as np
from tensorflow.keras.optimizers import Adam

from config import config
from loader import DataLoader
from networks import ProGAN
from networks.utils import plot_progan
from utils import logger
from utils.image_operations import plot_final_gif

N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)
N_BLOCKS = 7
BATCH_SIZE = [64, 64, 64, 64, 32, 16, 14]
GRADIENT_ACC_STEPS = [1, 1, 1, 1, 1, 1, 1]

LATENT_SIZE = 512
PATH = f"progan-{LATENT_SIZE}"
TRAIN_STEPS = int(1e6)

warm_start = True
starting_from_block = 2

gradient_penalty_weight = 10.0
n_critic = 1
train_steps = TRAIN_STEPS * n_critic

lp_path = os.path.join("learning_progress", PATH)
Path(lp_path).mkdir(parents=True, exist_ok=True)

image_ratio = config.get("image_ratio")
image_width = max(IMAGE_SIZES) * image_ratio[0]
image_height = max(IMAGE_SIZES) * image_ratio[1]


gan = ProGAN(
    gradient_penalty_weight=gradient_penalty_weight,
    latent_size=LATENT_SIZE,
    channels=3,
    img_width=image_width,
    img_height=image_height,
)

optimizer = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99)

gan.build_models(
    optimizer=optimizer,
    gradient_accumulation_steps=GRADIENT_ACC_STEPS[0],
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
    plot_progan(gan, lp_path, str(image_size))

    batch_size = BATCH_SIZE[block]
    gan.train(
        data_loader=data_loader,
        block=block,
        global_steps=TRAIN_STEPS,
        batch_size=batch_size,
        verbose=True,
        n_critic=n_critic,
        path=lp_path,
        write_model_to=model_dump_path,
    )
    gan.save(model_dump_path)
    gan.add_block(
        optimizer=optimizer,
        gradient_accumulation_steps=GRADIENT_ACC_STEPS[block],
    )

plot_final_gif(path=lp_path)
