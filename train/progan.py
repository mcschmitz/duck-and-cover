import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from AdamAcc import AdamAcc

from Loader.cover_loader import ImageLoader
from networks import ProGAN
from networks.utils import save_gan, load_gan, plot_gan
from utils import create_dir, generate_images

PATH = "2_progan"
FADE = [True, False]
WARM_START = [False, True]

RESOLUTIONS = 2 ** np.arange(2, 9)
BATCH_SIZE = 128
ACCUMULATIVE_UPDATES = {r: i for r, i in zip(RESOLUTIONS, [1, 1, 2, 4, 8, 16, 32])}
MINIBATCH_REPS = 4
GRADIENT_PENALTY_WEIGHT = 10
N_CRITIC = 1
TRAIN_STEPS = int(10e5)
IMAGE_RATIO = (1, 1)

if __name__ == "__main__":
    init_burn_in = False

    for resolution, fade, warm_start in itertools.product(RESOLUTIONS[4:], FADE, WARM_START):
        if init_burn_in and resolution == 4 or resolution > 4 and not warm_start:
            continue
        if resolution == 4:
            print("\n\nStarting init burn in with resolution {}\n\n".format(resolution))
            fade = False
            warm_start = False
        warm_start = True if fade else warm_start

        if fade:
            print("\n\nStarting fade in with resolution {}\n\n".format(resolution))
        elif resolution > 4:
            print("\n\nStarting burn in with resolution {}\n\n".format(resolution))

        DATA_PATH = "data/all_covers/all{}.npy".format(resolution)

        images = None

        lp_path = create_dir("learning_progress/{}".format(PATH))
        model_dump_path = create_dir(os.path.join(lp_path, "model{}".format(resolution)))
        model_load_path = (
            model_dump_path if warm_start and not fade else os.path.join(lp_path, "model{}".format(resolution // 2))
        )

        file_path_column = "file_path_{}".format(64 if resolution <= 64 else 256)
        covers = pd.read_json("../data/album_data_frame.json", orient="records", lines=True)
        covers.dropna(subset=[file_path_column], inplace=True)
        covers.reset_index(inplace=True)
        data_loader = ImageLoader(
            data=covers, path_column=file_path_column, image_size=resolution, image_ratio=IMAGE_RATIO
        )

        if os.path.exists(DATA_PATH) and os.stat(DATA_PATH).st_size < (psutil.virtual_memory().total * 0.8):
            images = np.load(DATA_PATH)
            img_idx = np.arange(0, images.shape[0])
        elif os.path.exists(DATA_PATH):
            pass
        else:
            try:
                images = data_loader.load_all()
                np.save(DATA_PATH, images)
                img_idx = np.arange(0, images.shape[0])
            except MemoryError as ex:
                print("Data does not fit inside Memory. Preallocation is not possible.")

        optimizer = AdamAcc(0.001, beta_1=0.0, beta_2=0.99, epsilon=10e-8, iters=ACCUMULATIVE_UPDATES[resolution])
        minibatch_size = BATCH_SIZE * N_CRITIC
        img_resolution = resolution if not fade else resolution // 2
        gan = ProGAN(
            batch_size=RESOLUTIONS[-1], image_resolution=img_resolution, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT
        )
        gan.build_models(optimizer=optimizer)
        plot_gan(gan, lp_path, str(resolution) + "_build")

        batch_idx = 0
        steps = TRAIN_STEPS // BATCH_SIZE
        initial_iter = 0

        if warm_start or fade:
            gan = load_gan(gan, model_load_path)

            if fade:
                gan.add_fade_in_layers(target_resolution=resolution)
                gan.build_models(optimizer=optimizer, rewire=True)
                plot_gan(gan, lp_path, str(resolution) + "_fade_in")
                alphas = np.linspace(0, 1, steps).tolist()

        for step in range(initial_iter // BATCH_SIZE, steps):
            if fade:
                alpha = alphas.pop(0)
                gan.update_alpha(alpha)

            if images is not None:
                batch_idx = [
                    i if i < images.shape[0] else i - images.shape[0]
                    for i in np.arange(batch_idx, batch_idx + minibatch_size)
                ]
                if 0 in batch_idx and images.shape[0] in batch_idx:
                    np.random.shuffle(img_idx)
                    images = images[img_idx]
                batch_images = images[batch_idx]
                batch_idx = batch_idx[-1] + 1
            else:
                batch_images = data_loader.next(batch_size=minibatch_size)

            for _ in range(MINIBATCH_REPS):
                gan.train_on_batch(batch_images, n_critic=N_CRITIC)
            gan.images_shown += RESOLUTIONS[-1]

            if step % (steps // 10) == 0:
                print(
                    "Images shown {0}: Generator Loss: {1:3,.3f} - Discriminator Loss + : {2:3,.3f}"
                    " Discriminator Loss - : {3:3,.3f} - Discriminator Loss Dummies : {4:3,.3f}".format(
                        gan.images_shown,
                        np.mean(gan.history["G_loss"]),
                        np.mean(gan.history["D_loss_positives"]),
                        np.mean(gan.history["D_loss_negatives"]),
                        np.mean(gan.history["D_loss_dummies"]),
                    )
                )

                generate_images(
                    gan.generator,
                    os.path.join(lp_path, "step{}.png".format(gan.images_shown)),
                    target_size=(RESOLUTIONS[-1] * 10, RESOLUTIONS[-1]),
                )
                generate_images(
                    gan.generator,
                    os.path.join(lp_path, "fixed_step{}.png".format(gan.images_shown)),
                    target_size=(RESOLUTIONS[-1] * 10, RESOLUTIONS[-1]),
                    seed=101,
                )

                x_axis = np.linspace(0, gan.images_shown, len(gan.history["D_loss_positives"]))
                ax = sns.lineplot(x_axis, gan.history["D_loss_positives"])
                plt.ylabel("Discriminator Loss on Positives")
                plt.xlabel("Images shown")
                plt.savefig(os.path.join(lp_path, "d_lossP.png"))
                plt.close()

                ax = sns.lineplot(x_axis, gan.history["D_loss_negatives"])
                plt.ylabel("Discriminator Loss on Negatives")
                plt.xlabel("Images shown")
                plt.savefig(os.path.join(lp_path, "d_lossN.png"))
                plt.close()

                ax = sns.lineplot(x_axis, gan.history["D_loss_dummies"])
                plt.ylabel("Discriminator Loss on Dummies")
                plt.xlabel("Images shown")
                plt.savefig(os.path.join(lp_path, "d_lossD.png"))
                plt.close()

                ax = sns.lineplot(x_axis, gan.history["G_loss"])
                plt.ylabel("Generator Loss")
                plt.xlabel("Images shown")
                plt.savefig(os.path.join(lp_path, "g_loss.png"))
                plt.close()

        if fade:
            gan.remove_fade_in_layers()
            gan.build_models(optimizer=optimizer, rewire=True)
        save_gan(gan, model_dump_path)

        plot_gan(gan, lp_path, str(resolution) + "_grown")

        if resolution == 4:
            init_burn_in = True
