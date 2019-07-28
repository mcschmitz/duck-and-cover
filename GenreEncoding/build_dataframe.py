import json

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from collect_artist_data import ARTIST_DATA_PATH

ARTIST_GENRE_PATH = "data/artist_data/artist_genre.h5"
ARTIST_GENRE_CORR_PATH = "data/artist_data/artist_genre_correlation.h5"

if __name__ == "__main__":
    with open(ARTIST_DATA_PATH, "r", encoding="utf-8") as file:
        artists = json.load(file)
        file.close()

    artist_genres = {a: artists[a]["genres"] for a in artists}

    genres = [artist_genres[a] for a in artist_genres]
    ids = [a for a in artist_genres]

    mlb = MultiLabelBinarizer()
    mlb.fit(genres)

    df = pd.DataFrame(mlb.transform(genres), columns=mlb.classes_, index=ids)

    df.to_hdf(ARTIST_GENRE_PATH, index=False, key="df", format="fixed", complib="blosc:lz4", complevel=9)

    correlation = df.corr()
    correlation.to_hdf(ARTIST_GENRE_CORR_PATH, index=False, key="df", format="fixed", complib="blosc:lz4", complevel=9)
