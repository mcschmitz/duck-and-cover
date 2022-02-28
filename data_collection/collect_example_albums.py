import pandas as pd
import spotipy
from dotenv import load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer
from spotipy.oauth2 import SpotifyClientCredentials

from data_collection import (
    ALBUM_DATA_PATH,
    GENRES_PATH,
    TEST_DATA_ALBUM_IDS,
    TEST_DATA_PATH,
)

meta_df = pd.read_json(ALBUM_DATA_PATH, orient="records", lines=True)
label_binarizer = MultiLabelBinarizer()
label_binarizer.fit(meta_df["artist_genre"].to_list())

load_dotenv()
spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials()
)


def get_album_info(album_id: str):
    """
    Collects the necessary information to create test album covers similar to a
    real cover.

    Args:
        album_id: Spotify album ID.
    """
    if album_id in meta_df["album_id"]:
        raise ValueError(f"album_id {album_id} in train data")

    album = spotify.album(album_id)
    album_name = album["name"]
    album_release = int(album["release_date"][:4])
    artist = album["artists"][0]
    artist_id = artist["id"]
    artist_name = artist["name"]
    artist_genre = spotify.artist(artist_id)["genres"]
    binarized_labels = label_binarizer.transform([artist_genre])
    binarized_labels = binarized_labels.squeeze()
    binarized_classes = label_binarizer.classes_[binarized_labels.astype(bool)]
    missing_genres = set(artist_genre).difference(set(binarized_classes))
    return {
        "artist_id": artist_id,
        "artist_name": artist_name,
        "artist_genre": artist_genre,
        "album_id": album_id,
        "album_name": album_name,
        "album_release": album_release,
        "missing_genres": missing_genres,
    }


if __name__ == "__main__":
    with open(TEST_DATA_ALBUM_IDS, "r", encoding="utf-8") as test_file:
        album_ids = [line.rstrip("\n") for line in test_file]

    album_infos = []
    for album_id in album_ids:
        album_infos.append(get_album_info(album_id))
    album_infos = pd.DataFrame(album_infos)
    album_infos.to_json(TEST_DATA_PATH, lines=True, orient="records")
    missing_genres = album_infos["missing_genres"].values
    missing_genres = {g for missing_set in missing_genres for g in missing_set}
    with open(GENRES_PATH, "r", encoding="utf-8") as genres_file:
        all_genres = [line.rstrip("\n") for line in genres_file]
    all_genres.extend(missing_genres)
    with open(GENRES_PATH, "w", encoding="utf-8") as genres_file:
        for f in all_genres:
            genres_file.write("%s\n" % f)
        genres_file.close()
