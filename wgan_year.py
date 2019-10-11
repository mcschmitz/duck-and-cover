import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from Loader.cover_loader import ImageLoader
from Networks import WGAN
from utils import create_dir, generate_images

BATCH_SIZE = 16
EPOCH_NUM = 100
PATH = "2_wgan_year"
DISCRIMINATOR_TRAIN_RATIO = 5
image_size = 16
image_ratio = (1, 1)

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)
year_scaler = StandardScaler()
year_scaler.fit(covers["album_release"].values.reshape(-1, 1))

data_loader = ImageLoader(data=covers, root="data/covers", batch_size=BATCH_SIZE, image_size=image_size,
                          image_ratio=image_ratio)
steps_per_epoch = len(covers) // BATCH_SIZE

image_width = data_loader.image_shape[0]
image_height = data_loader.image_shape[1]

year_wgan = WGAN(img_width=image_width, img_height=image_height, latent_size=128, batch_size=BATCH_SIZE,
                 gradient_penalty_weight=10)
year_wgan.build_models(year=True)
d_loss_positives = []
d_loss_negatives = []
d_loss_dummies = []
g_loss = []

create_dir("learning_progress/{}".format(PATH))
i = 0
for epoch in range(0, EPOCH_NUM):
    for step in range(0, steps_per_epoch):
        step += 1
        images, original_year = data_loader.next(year=True)
        year = year_scaler.transform(original_year)
        cum_d_acc, cum_g_loss = year_wgan.train_on_batch(images, year=year, ratio=DISCRIMINATOR_TRAIN_RATIO)
        print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Loss + : {4:3,.3f} '
              'Discriminator Loss - : {5:3,.3f} - Discriminator Loss Dummies : {6:3,.3f}'.format(
            year_wgan.combined_model.n_epochs + 1, step, steps_per_epoch, cum_g_loss, cum_d_acc[0], cum_d_acc[1],
            cum_d_acc[2]))

        if step % 10000 == 0 or step == (steps_per_epoch - 1):
            generate_images(year_wgan.generator,
                            'learning_progress/{}/epoch{}_{}_{}.png'.format(
                                PATH, year_wgan.combined_model.n_epochs, step, str(original_year[0])),
                            year=int(year[0]))
    year_wgan.combined_model.n_epochs += 1
    year_wgan.reset_metrics()

    d_loss_positives.append(cum_d_acc[0])
    ax = sns.lineplot(range(year_wgan.combined_model.n_epochs), d_loss_positives)
    plt.ylabel('Discriminator Loss on Positives')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_lossP.png'.format(PATH))
    plt.close()

    d_loss_negatives.append(cum_d_acc[1])
    ax = sns.lineplot(range(year_wgan.combined_model.n_epochs), d_loss_positives)
    plt.ylabel('Discriminator Loss on Negatives')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_lossN.png'.format(PATH))
    plt.close()

    d_loss_dummies.append(cum_d_acc[2])
    ax = sns.lineplot(range(year_wgan.combined_model.n_epochs), d_loss_positives)
    plt.ylabel('Discriminator Loss on Dummies')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/d_lossD.png'.format(PATH))
    plt.close()

    g_loss.append(cum_g_loss)
    ax = sns.lineplot(range(year_wgan.combined_model.n_epochs), g_loss)
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/{}/g_loss.png'.format(PATH))
    plt.close()
