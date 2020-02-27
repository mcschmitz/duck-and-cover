"""
Trains the ProGAN.
"""

import os
import numpy as np
from keras.optimizers import Adam
from networks import ProGAN
from networks.utils import plot_progan, save_gan, load_progan
from utils import create_dir, load_data
import itertools

N_BLOCKS = 6
RESOLUTIONS = 2 ** np.arange(2, N_BLOCKS + 2)
LATENT_SIZE = RESOLUTIONS[-1]
DATA_PATH = "data/celeba"

PATH = "celeba/2_progan_1"
FADE = [True, False]
WARM_START = False

BATCH_SIZE = [128, 128, 64, 32, 16, 8, 4]
MINIBATCH_REPS = 1
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC = 1
TRAIN_STEPS = int(10e5)


if __name__ == "__main__":
    optimizer = Adam(0.001, beta_1=0.0, beta_2=0.99)

    gan = ProGAN(gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT, latent_size=LATENT_SIZE)
    gan.build_models(optimizer=optimizer, n_blocks=N_BLOCKS, channels=3, batch_size=BATCH_SIZE)

    for block, fade in itertools.product(range(0, N_BLOCKS), FADE):
        if fade and block == 0:
            continue
        resolution = RESOLUTIONS[block]
        print("\n\nStarting training for resolution {}\n\n".format(resolution))
        images, img_idx = load_data(DATA_PATH, size=resolution)

        batch_size = BATCH_SIZE[block]
        minibatch_size = batch_size * N_CRITIC

        lp_path = os.path.join("learning_progress", PATH)
        model_dump_path = create_dir(os.path.join(lp_path, "model{}".format(resolution)))
        if WARM_START:
            previous_model_path = os.path.join(lp_path, "model{}".format(resolution // 2))
            gan = load_progan(gan, previous_model_path)
        plot_progan(gan, block, lp_path, str(resolution))

        steps = TRAIN_STEPS // batch_size
        alphas = np.linspace(0, 1, steps).tolist()

        gan.train(
            x=images,
            idx=img_idx,
            block=block,
            steps=steps,
            batch_size=batch_size,
            fade=fade,
            verbose=True,
            minibatch_size=minibatch_size,
            minibatch_reps=MINIBATCH_REPS,
            n_critic=N_CRITIC,
            path=lp_path,
        )
        save_gan(gan, model_dump_path)
