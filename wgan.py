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
from Networks import WGAN
from Networks.utils import save_gan, load_cover_gan
from utils import create_dir, generate_images, AnimatedGif

BATCH_SIZE = 32
EPOCH_NUM = 100
PATH = "1_wgan"
WARM_START = True
DATA_PATH = "data/all_covers/all64.npy"

image_size = 64
image_ratio = (1, 1)
images = None
gradient_penalty_weight = 10

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
minibatch_size = BATCH_SIZE * gradient_penalty_weight
steps_per_epoch = len(covers) // minibatch_size

gan = WGAN(img_width=image_width, img_height=image_height, latent_size=128, batch_size=BATCH_SIZE,
           gradient_penalty_weight=gradient_penalty_weight)
gan.build_models(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9))

if WARM_START:
    gan = load_cover_gan(gan, model_path)

batch_idx = 0
for epoch in range(gan.n_epochs, EPOCH_NUM):
    for step in range(0, steps_per_epoch):
        step += 1
        if images:
            batch_idx = [i if i < images.shape[0] else i - images.shape[0] for i in
                         np.arange(batch_idx, batch_idx + minibatch_size)]
            batch_images = images[batch_idx]
            batch_idx = batch_idx[-1] + 1
        else:
            batch_images = data_loader.next(batch_size=minibatch_size)
        cum_d_losses, cum_g_loss = gan.train_on_batch(batch_images)
        if step % 100 == 0 or step == steps_per_epoch:
            print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Loss + : {4:3,.3f} '
                  'Discriminator Loss - : {5:3,.3f} - Discriminator Loss Dummies : {6:3,.3f}'.format(
                gan.n_epochs + 1, step, steps_per_epoch, cum_g_loss, cum_d_losses[0], cum_d_losses[1],
                cum_d_losses[2]))
    gan.history["D_loss_positives"].append(cum_d_losses[0])
    gan.history["D_loss_negatives"].append(cum_d_losses[1])
    gan.history["D_loss_dummies"].append(cum_d_losses[2])
    gan.history["G_loss"].append(cum_g_loss)
    if images:
        np.random.shuffle(img_idx)
        images = images[img_idx]

    generate_images(gan.generator, os.path.join(lp_path, "epoch{}.png".format(gan.n_epochs)))
    generate_images(gan.generator, os.path.join(lp_path, "fixed_epoch{}.png".format(gan.n_epochs)), fixed=True)
    gan.n_epochs += 1
    gan.reset_metrics()

    ax = sns.lineplot(range(gan.n_epochs), gan.history["D_loss_positives"])
    plt.ylabel('Discriminator Loss on Positives')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_lossP.png'.format(PATH))
    plt.close()

    ax = sns.lineplot(range(gan.n_epochs), gan.history["D_loss_negatives"])
    plt.ylabel('Discriminator Loss on Negatives')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_lossN.png'.format(PATH))
    plt.close()

    ax = sns.lineplot(range(gan.n_epochs), gan.history["D_loss_dummies"])
    plt.ylabel('Discriminator Loss on Dummies')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_lossD.png'.format(PATH))
    plt.close()

    ax = sns.lineplot(range(gan.n_epochs), gan.history["G_loss"])
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/g_loss.png'.format(PATH))
    plt.close()
    save_gan(gan, model_path)

animated_gif = AnimatedGif(size=(image_width * 10, image_height + 50))
images = []
labels = []
for root, dirs, files in os.walk(lp_path):
    for file in files:
        if 'fixed_epoch' in file:
            images.append(imageio.imread(os.path.join(root, file)))
            labels.append(int(re.findall("\d+", file)[0]))

order = np.argsort(labels)
images = [images[i] for i in order]
labels = [labels[i] for i in order]
for img, lab in zip(images, labels):
    animated_gif.add(img, label="Epoch {}".format(lab))
animated_gif.save(os.path.join(lp_path, 'fixed.gif'))
