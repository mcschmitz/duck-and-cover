import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Loader.cover_loader import ImageLoader
from Networks import CoverGAN
from Networks.utils import save_gan, load_cover_gan, PixelNorm
from utils import create_dir, generate_images

BATCH_SIZE = 32
EPOCH_NUM = 1
PATH = "0_gan"
WARM_START = False
image_size = 32
image_ratio = (1, 1)

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)
lp_path = create_dir("learning_progress/{}".format(PATH))
model_path = create_dir(os.path.join(lp_path, "model"))

data_loader = ImageLoader(data=covers, root="data/covers", batch_size=BATCH_SIZE, image_size=image_size,
                          image_ratio=image_ratio)
steps_per_epoch = len(covers) // BATCH_SIZE

image_width = data_loader.image_shape[0]
image_height = data_loader.image_shape[1]

if not WARM_START:
    gan = CoverGAN(img_width=image_width, img_height=image_height, latent_size=128)
    gan.build_models()
    d_acc = []
    g_loss = []
    i = 0

else:
    custom_objects = {"PixelNorm": PixelNorm}
    gan = load_cover_gan(model_path, custom_objects)

for epoch in range(0, EPOCH_NUM):
    for step in range(0, steps_per_epoch):
        step += 1
        images = data_loader.next()
        cum_d_acc, cum_g_loss = gan.train_on_batch(images)
        print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Acc.: {4:3,.3f}'.format(
            gan.n_epochs, step, steps_per_epoch, cum_g_loss, cum_d_acc))
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
