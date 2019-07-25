import keras.backend as K
import pandas as pd
import seaborn as sns
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import load_model
from keras.optimizers import Adadelta
from keras.regularizers import l1

# General Parameters
GENRE_DF_PATH = "data/GenreEncoder/genre_data.h5"
genre_df = pd.read_hdf(GENRE_DF_PATH, "df")

# TSNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, init="pca", metric="jaccard", perplexity=50, learning_rate=100)
embedding2 = tsne.fit_transform(genre_df)
np.save("GenreEncoder/tsne/embedding2.npy", embedding2)

sns.scatterplot(embedding2[:, 0], embedding2[:, 1])

# EASE
import numpy as np


def ease(x, l):
    g = x.dot(x.T)
    diag_indices = np.diag_indices(g.shape[0])
    g[diag_indices] += l
    p = np.linalg.inv(g)
    b = p / (-np.diag(p))
    b[diag_indices] = 0
    return b


b = ease(genre_df.values, 3000)

BATCH_SIZE = 256
EPOCHS = 150
VALIDATION_SPLIT = .125


