import json

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from collect_artist_data import ARTIST_DATA_PATH

with open(ARTIST_DATA_PATH, "r", encoding="utf-8") as file:
    artists = json.load(file)

artist_genres = {a: artists[a]["genres"] for a in artists}

genres = [artist_genres[a] for a in artist_genres]

mlb = MultiLabelBinarizer()
mlb.fit(genres)

df = pd.DataFrame(mlb.transform(genres), columns=mlb.classes_)

df.to_hdf("data/GenreEncoder/genre_data.h5", index=False, key="df", format="fixed", complib="blosc:lz4", complevel=9)

correlation = df.corr()
correlation.to_hdf("data/GenreEncoder/genre_correlation.h5", index=False, key="df", format="fixed", complib="blosc:lz4",
                   complevel=9)
