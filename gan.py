import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Loader.cover_loader import ImageLoader
from Networks import CoverGAN
from Networks.utils import save_gan, load_cover_gan, PixelNorm
from utils import create_dir, generate_images

BATCH_SIZE = 32
EPOCH_NUM = 100
PATH = "0_gan"
WARM_START = False
DATA_PATH = "data/covers/all32.npy"
image_size = 32
image_ratio = (1, 1)

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)
lp_path = create_dir("learning_progress/{}".format(PATH))
model_path = create_dir(os.path.join(lp_path, "model"))

if not os.path.exists(DATA_PATH):
    data_loader = ImageLoader(data=covers, root="data/covers", batch_size=BATCH_SIZE, image_size=image_size,
                              image_ratio=image_ratio)
    images = data_loader.load_all()
    np.save(DATA_PATH, images)
else:
    images = np.load(DATA_PATH)

image_width = images.shape[1]
image_height = images.shape[2]
steps_per_epoch = len(covers) // BATCH_SIZE

if not WARM_START:
    gan = CoverGAN(img_width=image_width, img_height=image_height, latent_size=128)
    gan.build_models()
    d_acc = []
    g_loss = []
    i = 0

else:
    custom_objects = {"PixelNorm": PixelNorm}
    gan = load_cover_gan(model_path, custom_objects)

batch_idx = 0
img_idx = np.arange(0, images.shape[0])
for epoch in range(0, EPOCH_NUM):
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
    np.random.shuffle(img_idx)
    images = images[img_idx]
    generate_images(gan.generator, os.path.join(lp_path, "epoch{}.png".format(gan.n_epochs)))
    gan.n_epochs += 1
    gan.reset_metrics()

    d_acc.append(cum_d_acc)
    ax = sns.lineplot(range(gan.n_epochs), d_acc)
    plt.ylabel('Discriminator Accucary')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(lp_path, "d_acc.png".format(PATH)))
    plt.close()

    g_loss.append(cum_g_loss)
    ax = sns.lineplot(range(gan.n_epochs), g_loss)
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(lp_path, "g_loss.png".format(PATH)))
    plt.close()
    save_gan(gan, model_path)
