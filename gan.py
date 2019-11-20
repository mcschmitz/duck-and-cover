import os
import re

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from keras.optimizers import Adam

from Loader.cover_loader import ImageLoader
from Networks import CoverGAN
from Networks.utils import save_gan, load_cover_gan
from utils import create_dir, generate_images, AnimatedGif

BATCH_SIZE = 128
PATH = "0_gan"
WARM_START = True
DATA_PATH = "data/all_covers/all64.npy"
TRAIN_STEPS = int(10e6)

image_size = 64
image_ratio = (1, 1)
images = None

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)
covers.dropna(subset=["file_path_64"], inplace=True)
covers.reset_index(inplace=True)
lp_path = create_dir("learning_progress/{}".format(PATH))
model_path = create_dir(os.path.join(lp_path, "model"))
data_loader = ImageLoader(data=covers, path_column="file_path_64", image_size=image_size, image_ratio=image_ratio)

if os.path.exists(DATA_PATH) and os.stat(DATA_PATH).st_size < (psutil.virtual_memory().total * .8):
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

image_width = image_size * image_ratio[0]
image_height = image_size * image_ratio[1]

gan = CoverGAN(img_width=image_width, img_height=image_height, latent_size=128)
gan.build_models(combined_optimizer=Adam(0.0002, 0.5), discriminator_optimizer=Adam(0.000004))

if WARM_START:
    gan = load_cover_gan(gan, model_path)

batch_idx = 0
steps = TRAIN_STEPS // BATCH_SIZE
for step in range(gan.images_shown // BATCH_SIZE, steps):
    if images:
        batch_idx = [i if i < images.shape[0] else i - images.shape[0] for i in
                     np.arange(batch_idx, batch_idx + BATCH_SIZE)]
        if 0 in batch_idx and images.shape[0] in batch_idx:
            np.random.shuffle(img_idx)
            images = images[img_idx]
        batch_images = images[batch_idx]
        batch_idx = batch_idx[-1] + 1
    else:
        batch_images = data_loader.next()
    gan.train_on_batch(batch_images)
    gan.images_shown += BATCH_SIZE

    if step % 250 == 0:
        print('Images shown {0}: Generator Loss: {1:2,.3f} - Discriminator Acc.: {2:3,.3f}'.format(
            gan.images_shown, np.mean(gan.history["G_loss"]), np.mean(gan.history["D_accuracy"])))

        generate_images(gan.generator, os.path.join(lp_path, "step{}.png".format(gan.images_shown)))
        generate_images(gan.generator, os.path.join(lp_path, "fixed_step{}.png".format(gan.images_shown)), fixed=True)

        ax = sns.lineplot(np.linspace(0, gan.images_shown, len(gan.history["D_accuracy"])), gan.history["D_accuracy"])
        plt.ylabel('Discriminator Accucary')
        plt.xlabel('Images shown')
        plt.savefig(os.path.join(lp_path, "d_acc.png".format(PATH)))
        plt.close()

        ax = sns.lineplot(np.linspace(0, gan.images_shown, len(gan.history["G_loss"])), gan.history["G_loss"])
        plt.ylabel('Generator Loss')
        plt.xlabel('Images shown')
        plt.savefig(os.path.join(lp_path, "g_loss.png".format(PATH)))
        plt.close()

        save_gan(gan, model_path)

animated_gif = AnimatedGif(size=(image_width * 10, image_height + 50))
images = []
labels = []
for root, dirs, files in os.walk(lp_path):
    for file in files:
        if 'fixed_step' in file:
            images.append(imageio.imread(os.path.join(root, file)))
            labels.append(int(re.findall("\d+", file)[0]))

order = np.argsort(labels)
images = [images[i] for i in order]
labels = [labels[i] for i in order]
for img, lab in zip(images, labels):
    animated_gif.add(img, label="{} Images shown".format(lab))
animated_gif.save(os.path.join(lp_path, 'fixed.gif'))
