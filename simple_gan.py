import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Loader.cover_loader import ImageLoader, array_to_img, rescale_images
from Networks.simple_gan import DaCSimple
from utils import create_dir

BATCH_SIZE = 128
EPOCH_NUM = 1000
image_size = 64
image_ratio = (1, 1)

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)

steps_per_epoch = len(covers) // BATCH_SIZE

data_loader = ImageLoader(data=covers, root="data/covers", batch_size=BATCH_SIZE, image_size=image_size,
                          image_ratio=image_ratio)

image_width_compr = data_loader.image_shape[0]
image_height_compr = data_loader.image_shape[1]

dac = DaCSimple(img_width=image_width_compr, img_height=image_height_compr, latent_size=512)
d_acc = []
g_loss = []

create_dir("learning_progress/simple_gan")
for epoch in range(0, EPOCH_NUM):
    for step in range(0, steps_per_epoch):
        step += 1
        real_images = data_loader.next(year=False, genre=False)
        cum_d_acc, cum_g_loss = dac.train_on_batch(real_images)
        print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Acc.: {4:3,.3f}'.format(
            dac.adversarial_model.n_epochs + 1, step, steps_per_epoch, cum_g_loss, cum_d_acc))

        if step % 1000 == 0:
            noise = np.random.uniform(size=(1, dac.latent_size))
            img = dac.generator.predict([noise])[0]
            img = rescale_images(img)
            img = array_to_img(img)
            prediction_path = 'learning_progress/simple_gan/epoch{}_{}.png'.format(dac.adversarial_model.n_epochs, step)
            img.save(prediction_path)
    dac.adversarial_model.n_epochs += 1
    dac.reset_metrics()

    d_acc.append(cum_d_acc)
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), d_acc)
    plt.ylabel('Discriminator Accucary')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/simple_gan/d_acc.png')
    plt.close()

    g_loss.append(cum_g_loss)
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), g_loss)
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/simple_gan/g_loss.png')
    plt.close()
