import pandas as pd
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import *
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l1
import numpy as np


# General Parameters
GENRE_DF_PATH = "data/GenreEncoder/genre_data.h5"
BATCH_SIZE = 256
EPOCHS = 150
VALIDATION_SPLIT = .125

genre_df = pd.DataFrame(pd.read_hdf(GENRE_DF_PATH, "df"))
active = genre_df.values.mean()

eta = 1 - genre_df.values.mean(axis=0)


def weighted_binary_crossentropy(y_true, y_pred):
    binary_cross_entroy = K.binary_crossentropy(y_true, y_pred)
    weighted_v = (y_true ** eta) * ((1 - y_true) ** (1 - eta))
    weighted_bce = weighted_v * binary_cross_entroy
    return K.mean(weighted_bce, axis=-1)


genre_input = Input(shape=(genre_df.shape[1],))
enc = Dense(1024, activation="relu")(genre_input)
enc = Dense(512, activation="relu")(enc)
encoded_genre = Dense(50, activation="relu", activity_regularizer=l1(10e-5))(enc)

dec = Dense(512, activation="relu")(encoded_genre)
dec = Dense(1024, activation="relu")(dec)
decoded_genre = Dense(genre_df.shape[1], use_bias=False, activation="sigmoid")(dec)

autoencoder = Model(genre_input, decoded_genre)
autoencoder.compile(optimizer=Adam(10e-6), loss=[weighted_binary_crossentropy])

model_checkpoint = ModelCheckpoint(filepath="GenreEncoder/autoencoder/autoencoder.hdf5", save_best_only=True)
early_stopping = EarlyStopping(patience=EPOCHS//10)

autoencoder.fit(genre_df.values, genre_df.values, epochs=EPOCHS, batch_size=BATCH_SIZE,
                callbacks=[model_checkpoint, early_stopping],
                validation_split=VALIDATION_SPLIT, shuffle=True)

autoencoder = load_model(filepath="GenreEncoder/autoencoder/autoencoder.hdf5")

pred = autoencoder.predict(genre_df.values[:5])
pred.max(axis=1)
np.where(genre_df.values[:5] == 1)
np.where(np.round(pred) == 1)
