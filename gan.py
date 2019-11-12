import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Loader.cover_loader import ImageLoader
from Networks import CoverGAN
from Networks.utils import save_gan, load_cover_gan
from utils import create_dir, generate_images

BATCH_SIZE = 32
EPOCH_NUM = 100
PATH = "0_gan"
WARM_START = False
DATA_PATH = "data/all_covers/all64.npy"
image_size = 64
image_ratio = (1, 1)

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)
covers.dropna(subset=["file_path_64"], inplace=True)
covers.reset_index(inplace=True)
lp_path = create_dir("learning_progress/{}".format(PATH))
model_path = create_dir(os.path.join(lp_path, "model"))

if not os.path.exists(DATA_PATH):
    data_loader = ImageLoader(data=covers, path_column="file_path_64", batch_size=BATCH_SIZE, image_size=image_size,
                              image_ratio=image_ratio)
    images = data_loader.load_all()
    np.save(DATA_PATH, images)
else:
    images = np.load(DATA_PATH)

image_width = images.shape[1]
image_height = images.shape[2]
steps_per_epoch = len(covers) // BATCH_SIZE

gan = CoverGAN(img_width=image_width, img_height=image_height, latent_size=128)
gan.build_models()

if WARM_START:
    gan = load_cover_gan(gan, model_path)

batch_idx = 0
img_idx = np.arange(0, images.shape[0])
for epoch in range(gan.n_epochs, EPOCH_NUM):
    for step in range(0, steps_per_epoch):
        step += 1
        batch_idx = [i if i < images.shape[0] else i - images.shape[0] for i in
                     np.arange(batch_idx, batch_idx + BATCH_SIZE)]
        batch_images = images[batch_idx]
        batch_idx = batch_idx[-1] + 1
        cum_d_acc, cum_g_loss = gan.train_on_batch(batch_images)
        if step % 1000 == 0 or step == steps_per_epoch:
            print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Acc.: {4:3,.3f}'.format(
                gan.n_epochs, step, steps_per_epoch, cum_g_loss, cum_d_acc))
    gan.history["G_loss"].append(cum_g_loss)
    gan.history["D_accuracy"].append(cum_d_acc)
    np.random.shuffle(img_idx)
    images = images[img_idx]
    generate_images(gan.generator, os.path.join(lp_path, "epoch{}.png".format(gan.n_epochs)))
    generate_images(gan.generator, os.path.join(lp_path, "fixed_epoch{}.png".format(gan.n_epochs)), fixed=True)
    gan.n_epochs += 1
    gan.reset_metrics()

    ax = sns.lineplot(range(gan.n_epochs), gan.history["D_accuracy"])
    plt.ylabel('Discriminator Accucary')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(lp_path, "d_acc.png".format(PATH)))
    plt.close()

    ax = sns.lineplot(range(gan.n_epochs), gan.history["G_loss"])
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(lp_path, "g_loss.png".format(PATH)))
    plt.close()
    save_gan(gan, model_path)
