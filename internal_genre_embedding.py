import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

from Loader.internal_embedding.int_emb_loader import ImageLoader, array_to_img
from Networks.internal_embedding.int_emb_network import DACInternalGenreEmbedding

BATCH_SIZE = 32
EPOCH_NUM = 1000
image_size = 256
image_ratio = (1, 1)

covers = pd.read_json("data/album_data_frame.json", orient="records", lines=True)
covers = covers.sample(frac=1).reset_index(drop=True)

steps_per_epoch = len(covers) // BATCH_SIZE

mlb = MultiLabelBinarizer()
genre_prob = mlb.fit_transform(covers["artist_genre"].values.tolist()).mean()

data_loader = ImageLoader(data=covers, root="data/covers", binarizer=mlb, batch_size=BATCH_SIZE, image_size=image_size,
                          image_ratio=image_ratio)

image_width_compr = data_loader.image_shape[0]
image_height_compr = data_loader.image_shape[1]

dac = DACInternalGenreEmbedding(img_width=image_width_compr, img_height=image_height_compr, latent_size=64,
                                genre_prob=genre_prob, n_genres=len(mlb.classes_))
d_acc = []
g_loss = []

for epoch in range(0, EPOCH_NUM):
    for step in range(0, steps_per_epoch):
        step += 1
        real_images, genres = data_loader.next()
        cum_d_acc, cum_g_loss = dac.train_on_batch(real_images, genres)
        print('Epoch {0}: Batch {1}/{2} - Generator Loss: {3:3,.3f} - Discriminator Acc.: {4:3,.3f}'.format(
            dac.adversarial_model.n_epochs + 1, step, steps_per_epoch, cum_g_loss, cum_d_acc))
    dac.adversarial_model.n_epochs += 1

    d_acc.append(cum_d_acc)
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), d_acc)
    plt.ylabel('Discriminator Accucary')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/internal_embedding/d_acc.png')
    plt.close()

    g_loss.append(cum_g_loss)
    ax = sns.lineplot(range(dac.adversarial_model.n_epochs), g_loss)
    plt.ylabel('Generator Loss')
    plt.xlabel('Epochs')
    plt.savefig('learning_progress/internal_embedding/g_loss.png')
    plt.close()

    if epoch % 1 == 0:
        noise = np.random.uniform(size=(1, dac.latent_size))
        genre_noise = np.random.choice([0, 1], size=(1, dac.n_genres), p=[1 - dac.genre_prob, dac.genre_prob])
        img = dac.generator.predict([noise, genre_noise])[0]
        img = array_to_img(img)
        prediction_path = 'learning_progress/internal_embedding/epoch{}.png'.format(dac.adversarial_model.n_epochs)
        img.save(prediction_path)
