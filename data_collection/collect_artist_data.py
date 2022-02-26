import json
import os

import pandas as pd
from dotenv import load_dotenv
from spotipy import util as util

from data_collection import (
    ALBUM_DATA_PATH,
    ARTISTS_FILE,
    REMAINING_ARTISTS,
    REMAINING_GENRES_PATH,
    SpotifyInfoCollector,
)

if __name__ == "__main__":
    load_dotenv()
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
    token = util.oauth2.SpotifyClientCredentials(
        client_id, client_secret
    ).get_access_token()
    artist_info_collector = SpotifyInfoCollector(
        token, spotify_id=client_id, spotify_secret=client_secret
    )

    if os.path.isfile(REMAINING_GENRES_PATH):
        with open(ARTISTS_FILE, "r", encoding="utf-8") as file:
            genres_to_process = [line.rstrip("\n") for line in file]
        with open(ARTISTS_FILE, "r", encoding="utf-8") as file:
            artists_to_process = json.load(file)
            file.close()
            artist_info_collector = SpotifyInfoCollector(
                spotify_token=token,
                spotify_id=client_id,
                spotify_secret=client_secret,
                artist_genres_map=artists_to_process,
            )
    else:
        with open(ARTISTS_FILE, "r", encoding="utf-8") as file:
            genres_to_process = [line.rstrip("\n") for line in file]
    artists_to_process = artist_info_collector.get_top_artists_for_genre(
        genres=genres_to_process,
        result_path=ARTISTS_FILE,
        remaining_path=REMAINING_GENRES_PATH,
        save_on=100,
    )

    if os.path.isfile(REMAINING_ARTISTS):
        with open(REMAINING_ARTISTS, "r", encoding="utf-8") as file:
            artists_to_process = json.load(file)
            file.close()
        album_data = pd.read_json(
            ALBUM_DATA_PATH, orient="records", lines=True
        )
        artist_info_collector = SpotifyInfoCollector(
            token,
            spotify_id=client_id,
            spotify_secret=client_secret,
            cover_frame=album_data,
        )
    else:
        with open(ARTISTS_FILE, "r", encoding="utf-8") as file:
            artists_to_process = json.load(file)
            file.close()
    artist_info_collector.build_cover_data_frame(
        artists_to_process, 1000, ALBUM_DATA_PATH, REMAINING_ARTISTS
    )

    album_data = pd.read_json(ALBUM_DATA_PATH, orient="records", lines=True)
    artist_info_collector.cover_frame = album_data
    artist_info_collector.collect_album_cover(
        target_dir="../data/covers300", size=300
    )
    artist_info_collector.collect_album_cover(
        target_dir="../data/covers64", size=64
    )

    album_data = pd.read_json(ALBUM_DATA_PATH, orient="records", lines=True)
    artist_info_collector = SpotifyInfoCollector(
        token,
        spotify_id=client_id,
        spotify_secret=client_secret,
        cover_frame=album_data,
    )
    for size in (64, 300):
        artist_info_collector.cover_frame = (
            artist_info_collector.add_file_path_to_frame(
                target_dir=f"data/all_covers{size}"
            )
        )
    artist_info_collector.cover_frame.to_json(
        ALBUM_DATA_PATH, lines=True, orient="records"
    )
