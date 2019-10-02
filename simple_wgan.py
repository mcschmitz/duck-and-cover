import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Loader.cover_loader import ImageLoader
from Networks.simple_wgan import DaCSimpleWGan
from utils import create_dir, generate_images

BATCH_SIZE = 128
EPOCH_NUM = 100
PATH = "simple_wgan"
DISCRIMINATOR_TRAIN_RATIO = 5
image_size = 64
image_ratio = (1, 1)
all_files_path = "data/covers/all64.npy"

#  TODO Add Release Year information
#  TODO Add Genre Information
#  TODO Let GAN grow

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)

steps_per_epoch = len(covers) // BATCH_SIZE

data_loader = ImageLoader(data=covers, root="data/covers", batch_size=BATCH_SIZE, image_size=image_size,
                          image_ratio=image_ratio)

if not os.path.exists(all_files_path):
    X_train = data_loader.load_all(year=False, genre=False)
    np.save(all_files_path, X_train)
else:
    X_train = np.load(all_files_path)

image_width = data_loader.image_shape[0]
image_height = data_loader.image_shape[1]

dac = DaCSimpleWGan(img_width=image_width, img_height=image_height, latent_size=512, batch_size=BATCH_SIZE)
dac.build_models(gradient_penalty_weight=10)
d_loss_positives = []
d_loss_negatives = []
d_loss_dummies = []
g_loss = []

steps_per_epoch = len(X_train) // BATCH_SIZE

create_dir("learning_progress/{}".format(PATH))
i = 0
for epoch in range(0, EPOCH_NUM):
    np.random.shuffle(X_train)
    for step in range(0, steps_per_epoch):
        step += 1
        idx = np.arange(i, i + BATCH_SIZE)
        idx = [i if i < len(X_train) else i - len(X_train) for i in idx]
        images = X_train[idx]
        cum_d_acc, cum_g_loss = dac.train_on_batch(images, ratio=DISCRIMINATOR_TRAIN_RATIO)
        print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Loss + : {4:3,.3f} '
              'Discriminator Loss - : {5:3,.3f} - Discriminator Loss Dummies : {6:3,.3f}'.format(
            dac.adversarial_model.n_epochs + 1, step, steps_per_epoch, cum_g_loss, cum_d_acc[0], cum_d_acc[1],
            cum_d_acc[2]))
        i = idx[-1]

        if step % 1000 == 0 or step == (steps_per_epoch - 1):
            generate_images(dac.generator,
                            'learning_progress/{}/epoch{}_{}.png'.format(PATH, dac.adversarial_model.n_epochs, step))
    dac.adversarial_model.n_epochs += 1
    dac.reset_metrics()

    d_loss_positives.append(cum_d_acc[0])
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), d_loss_positives)
    plt.ylabel('Discriminator Loss on Positives')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_acc.png'.format(PATH))
    plt.close()

    d_loss_negatives.append(cum_d_acc[1])
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), d_loss_positives)
    plt.ylabel('Discriminator Loss on Negatives')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_acc.png'.format(PATH))
    plt.close()

    d_loss_dummies.append(cum_d_acc[2])
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), d_loss_positives)
    plt.ylabel('Discriminator Loss on Dummies')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_acc.png'.format(PATH))
    plt.close()

    g_loss.append(cum_g_loss)
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), g_loss)
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/g_loss.png'.format(PATH))
    plt.close()
