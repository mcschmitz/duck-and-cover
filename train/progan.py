import logging
import os

import numpy as np
from tensorflow.keras.optimizers import Adam

from constants import (
    BASE_DATA_PATH,
    LOG_DATETIME_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
)
from loader import DataLoader
from networks import ProGAN
from networks.utils import load_progan, plot_progan, save_gan
from tf_init import init_tf
from utils import create_dir, plot_final_gif

logging.basicConfig(
    format=LOG_FORMAT, datefmt=LOG_DATETIME_FORMAT, level=LOG_LEVEL
)
logger = logging.getLogger(__file__)

N_BLOCKS = 7
IMAGE_SIZES = 2 ** np.arange(2, N_BLOCKS + 2)
BATCH_SIZE = [16, 16, 16, 16, 16, 16, 14]
GRADIENT_ACC_STEPS = {
    64: [1, 1, 1, 1, 1, 1, 14],
    256: [1, 1, 1, 1, 1, 2, 14],
    1024: [1, 1, 1, 1, 1, 1, 7],
}
LATENT_SIZE = 1024
PATH = f"{LATENT_SIZE}_progan"
TRAIN_STEPS = int(1e6)
img_height = img_width = 2 ** (N_BLOCKS + 1)

gradient_penalty_weight = 10.0
n_critic = 1
train_steps = TRAIN_STEPS * n_critic
warm_start = True

starting_from_block = 6

lp_path = os.path.join("learning_progress", PATH)

init_tf()
gan = ProGAN(
    gradient_penalty_weight=gradient_penalty_weight,
    latent_size=LATENT_SIZE,
    channels=3,
    img_width=img_width,
    img_height=img_height,
)

optimizer = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99)
gan.build_models(
    optimizer=optimizer,
    n_blocks=N_BLOCKS,
    gradient_accumulation_steps=GRADIENT_ACC_STEPS[LATENT_SIZE],
)

for block in range(starting_from_block, N_BLOCKS):
    image_size = IMAGE_SIZES[block]
    model_dump_path = create_dir(os.path.join(lp_path, f"model{image_size}"))
    model_path = (
        model_dump_path
        if os.path.exists(model_dump_path + "/C_0_0.h5")
        else os.path.join(lp_path, "model{}".format(image_size // 2))
    )
    grad_acc_steps = GRADIENT_ACC_STEPS[LATENT_SIZE][block]
    data_path = os.path.join(
        BASE_DATA_PATH, f"covers{300 if image_size > 64 else 64}"
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
        grad_acc_steps=grad_acc_steps,
    )
    save_gan(gan, model_dump_path)

plot_final_gif(path=lp_path)
