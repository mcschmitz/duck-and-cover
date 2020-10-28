import os
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from keras.optimizers import Adam

from loader import DataLoader
from networks import WGAN
from networks import load_progan, save_gan
from utils import AnimatedGif, create_dir, generate_images

BATCH_SIZE = 64
PATH = "1_wgan"
WARM_START = False
DATA_PATH = "../data/covers64/all64.npy"
TRAIN_STEPS = int(2 * 10e5)

image_size = 64
image_ratio = (1, 1)
images = None
gradient_penalty_weight = 10
n_critic = 5

lp_path = create_dir("learning_progress/{}".format(PATH))
model_path = create_dir(os.path.join(lp_path, "model"))

if __name__ == "__main__":
    covers = pd.read_json(
        "../data/album_data_frame.json", orient="records", lines=True
    )
    covers.dropna(subset=["file_path_64"], inplace=True)
    covers.reset_index(inplace=True)
    data_loader = DataLoader(
        image_path=covers,
        path_column="file_path_64",
        image_size=image_size,
        image_ratio=image_ratio,
    )

    if os.path.exists(DATA_PATH) and os.stat(DATA_PATH).st_size < (
        psutil.virtual_memory().total * 0.8
    ):
        images = np.load(DATA_PATH)
        img_idx = np.arange(0, images.shape[0])
    elif os.path.exists(DATA_PATH):
        pass
    else:
        try:
            images = data_loader.load_all()
            np.save(DATA_PATH, images)
            img_idx = np.arange(0, images.shape[0])
        except MemoryError:
            print(
                "Data does not fit inside Memory. Preallocation is not possible."
            )

    image_width = image_size * image_ratio[0]
    image_height = image_size * image_ratio[1]
    minibatch_size = BATCH_SIZE * n_critic

    gan = WGAN(
        img_width=image_width,
        img_height=image_height,
        latent_size=128,
        batch_size=BATCH_SIZE,
        gradient_penalty_weight=gradient_penalty_weight,
    )
    gan.build_models(optimizer=Adam(0.0001, beta_1=0.5))

    if WARM_START:
        gan = load_progan(gan, model_path)

    batch_idx = 0
    steps = TRAIN_STEPS // BATCH_SIZE
    for step in range(0, steps):
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
            batch_images = data_loader.get_next_batch(
                batch_size=minibatch_size
            )
        gan.train_on_batch(batch_images)
        gan.images_shown += BATCH_SIZE

        if step % 250 == 0:
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
                target_size=(64 * 10, 64),
            )
            generate_images(
                gan.generator,
                os.path.join(
                    lp_path, "fixed_step{}.png".format(gan.images_shown)
                ),
                seed=True,
                target_size=(64 * 10, 64),
            )

            x_axis = np.linspace(
                0, gan.images_shown, len(gan.history["D_loss_positives"])
            )
            ax = sns.lineplot(x_axis, gan.history["D_loss_positives"])
            plt.ylabel("Discriminator Loss on Positives")
            plt.xlabel("Images shown")
            plt.savefig("learning_progress/{}/d_lossP.png".format(PATH))
            plt.close()

            ax = sns.lineplot(x_axis, gan.history["D_loss_negatives"])
            plt.ylabel("Discriminator Loss on Negatives")
            plt.xlabel("Images shown")
            plt.savefig("learning_progress/{}/d_lossN.png".format(PATH))
            plt.close()

            ax = sns.lineplot(x_axis, gan.history["D_loss_dummies"])
            plt.ylabel("Discriminator Loss on Dummies")
            plt.xlabel("Images shown")
            plt.savefig("learning_progress/{}/d_lossD.png".format(PATH))
            plt.close()

            ax = sns.lineplot(x_axis, gan.history["G_loss"])
            plt.ylabel("Generator Loss")
            plt.xlabel("Images shown")
            plt.savefig("learning_progress/{}/g_loss.png".format(PATH))
            plt.close()

            save_gan(gan, model_path)

    gif_size = (image_width * 10, image_height + 50)
    animated_gif = AnimatedGif(size=gif_size)
    images = []
    labels = []
    for root, dirs, files in os.walk(lp_path):
        for file in files:
            if "fixed_step" in file:
                images.append(imageio.imread(os.path.join(root, file)))
                labels.append(int(re.findall("\d+", file)[0]))

    order = np.argsort(labels)
    images = [images[i] for i in order]
    labels = [labels[i] for i in order]
    for img, lab in zip(images, labels):
        animated_gif.add(
            img,
            label="{} Images shown".format(lab),
            label_position=(gif_size[0] * 0.7, gif_size[1] * 0.7),
        )
    animated_gif.save(os.path.join(lp_path, "fixed.gif"), fps=len(images) / 30)
