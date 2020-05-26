import itertools
import os

import numpy as np
from keras.optimizers import Adam

from Loader import DataLoader
from networks import ProGAN
from networks.utils import load_progan, plot_progan, save_gan
from utils import create_dir

N_BLOCKS = 7
RESOLUTIONS = 2 ** np.arange(2, N_BLOCKS + 2)
RESOLUTIONS = RESOLUTIONS if isinstance(RESOLUTIONS, list) else RESOLUTIONS
LATENT_SIZE = 1024

PATH = "2_progan/base"
FADE = [True, False]
WARM_START = False

BATCH_SIZE = [128, 128, 64, 32, 16, 8, 4, 3]
MINIBATCH_REPS = 1
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC = 1
TRAIN_STEPS = int(10e5)

starting_from_block = 0

base_data_path = "/opt/input/data/covers{}"
# base_data_path = "data/covers{}"

optimizer = Adam(0.001, beta_1=0.0, beta_2=0.99)
gan = ProGAN(gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT, latent_size=LATENT_SIZE)
gan.build_models(optimizer=optimizer, n_blocks=N_BLOCKS, channels=3, batch_size=BATCH_SIZE)

for block, fade in itertools.product(range(0, 1), FADE):  # len(RESOLUTIONS)
    if fade and block == 0:
        continue
    resolution = RESOLUTIONS[block]
    block += starting_from_block
    print("\n\nStarting training for resolution {}\n\n".format(resolution))
    data_path = base_data_path.format(300 if resolution > 64 else 64)
    data_loader = DataLoader(data_path, image_size=resolution)

    batch_size = BATCH_SIZE[block]
    minibatch_size = batch_size * N_CRITIC

    lp_path = os.path.join("learning_progress", PATH)
    model_dump_path = create_dir(os.path.join(lp_path, "model{}".format(resolution)))
    if WARM_START:
        model_path = (
            model_dump_path
            if os.path.exists(model_dump_path + "/C_0_0.h5")
            else os.path.join(lp_path, "model{}".format(resolution // 2))
        )
        gan = load_progan(gan, model_path)
    plot_progan(gan, block, lp_path, str(resolution))

    print("\n\nStarting training for resolution {}\n\n".format(resolution))
    steps = TRAIN_STEPS // batch_size

    gan.train(
        data_loader=data_loader,
        block=block,
        steps=steps,
        batch_size=batch_size,
        fade=fade,
        verbose=True,
        minibatch_size=minibatch_size,
        minibatch_reps=MINIBATCH_REPS,
        n_critic=N_CRITIC,
        path=lp_path,
        write_model_to=model_dump_path,
    )
    save_gan(gan, model_dump_path)
