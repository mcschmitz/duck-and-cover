import os
from pathlib import Path

import numpy as np
from tensorflow.keras.optimizers import Adam

from config import config
from loader import DataLoader
from networks import ProGAN
from networks.utils import load_progan, plot_progan, save_gan
from utils import logger
from utils.image_operations import plot_final_gif

N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)
N_BLOCKS = 7
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14]
GRADIENT_ACC_STEPS = [1, 1, 1, 1, 1, 1, 1]

LATENT_SIZE = 512
PATH = f"progan-{LATENT_SIZE}"
TRAIN_STEPS = int(1e6)

warm_start = False
starting_from_block = 0

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

for block in range(starting_from_block, N_BLOCKS):
    gan.build_models(
        optimizer=optimizer,
        n_blocks=block,
        gradient_accumulation_steps=GRADIENT_ACC_STEPS,
    )
    image_size = IMAGE_SIZES[block]
    model_dump_path = os.path.join(lp_path, f"model-{image_size}x{image_size}")
    Path(model_dump_path).mkdir(parents=True, exist_ok=True)
    model_path = (
        model_dump_path
        if os.path.exists(model_dump_path + "/C_0_0.h5")
        else os.path.join(lp_path, "model{}".format(image_size // 2))
    )
    data_path = os.path.join(
        config.get("base_data_path"), f"covers{300 if image_size > 64 else 64}"
    )
    data_loader = DataLoader(data_path, image_size=image_size)
    batch_size = BATCH_SIZE[block]
    if warm_start:
        logger.info(f"Load model from {model_path}")
        gan = load_progan(gan, model_path)
    plot_progan(gan, block, lp_path, str(image_size))
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
    save_gan(gan, model_dump_path)

plot_final_gif(path=lp_path)
