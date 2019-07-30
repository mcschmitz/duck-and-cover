from datetime import datetime

import pandas as pd
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import *
from keras.models import load_model
from keras.optimizers import Adam
from keras.regularizers import l1
from sklearn.model_selection import train_test_split

from GenreEncoding.build_dataframe import ARTIST_GENRE_PATH
from GenreEncoding.keras_losses import jaccard_distance

# General Parameters
BATCH_SIZE = 64
EPOCHS = 1500
VALIDATION_SPLIT = .25

genre_df = pd.DataFrame(pd.read_hdf(ARTIST_GENRE_PATH, "df"))
train_set, valid_set = train_test_split(genre_df, test_size=VALIDATION_SPLIT)
valid_set, test_set = train_test_split(valid_set, test_size=VALIDATION_SPLIT)

genre_input = Input(shape=(genre_df.shape[1],), name="Input")
enc = Dense(1024, activation="relu", name="Encoder_Dense1", use_bias=False)(genre_input)
enc = Dense(512, activation="relu", name="Encoder_Dense2", use_bias=False)(enc)
encoded_genre = Dense(50, activation="relu", activity_regularizer=l1(10e-5), name="Encoder_out", use_bias=False)(enc)

dec = Dense(512, activation="relu", name="Decoder_Dense1", use_bias=False)(encoded_genre)
dec = Dense(1024, activation="relu", name="Decocer_Dense2", use_bias=False)(dec)
decoded_genre = Dense(genre_df.shape[1], use_bias=False, activation="sigmoid", name="Decoder_out")(dec)

autoencoder = Model(genre_input, decoded_genre)
autoencoder.compile(optimizer=Adam(10e-7), loss=[jaccard_distance])

model_checkpoint = ModelCheckpoint(filepath="GenreEncoding/autoencoder/autoencoder.hdf5", save_best_only=True)
early_stopping = EarlyStopping(patience=EPOCHS // 10)
now = datetime.now()
logdir = "GenreEncoding/autoencoder/tensorboard/" + now.strftime("%Y%m%d-%H%M%S") + "/"
tensor_board = TensorBoard(logdir, batch_size=BATCH_SIZE, histogram_freq=1)

autoencoder.fit(train_set.values, train_set.values, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_data=(valid_set.values, valid_set.values),
                callbacks=[model_checkpoint, early_stopping, tensor_board],
                validation_split=VALIDATION_SPLIT, shuffle=True)

autoencoder = load_model(filepath="GenreEncoding/autoencoder/autoencoder.hdf5")

pred = autoencoder.predict(test_set)
pred.max(axis=1)
np.where(test_set == 1)
np.where(np.round(pred) == 1)
