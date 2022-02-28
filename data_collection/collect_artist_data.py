import json
import os

import pandas as pd

from data_collection import (
    ALBUM_DATA_PATH,
    ARTISTS_FILE,
    GENRES_PATH,
    REMAINING_ARTISTS,
    REMAINING_GENRES_PATH,
    SpotifyInfoCollector,
)

if __name__ == "__main__":
    artist_info_collector = SpotifyInfoCollector()

    if os.path.isfile(REMAINING_GENRES_PATH):
        with open(REMAINING_GENRES_PATH, "r", encoding="utf-8") as rg_file:
            genres_to_process = [line.rstrip("\n") for line in rg_file]
        if os.path.isfile(ARTISTS_FILE):
            with open(ARTISTS_FILE, "r", encoding="utf-8") as a_file:
                artists_to_process = json.load(a_file)
                a_file.close()
            artist_info_collector.artist_genres = artists_to_process
    else:
        with open(GENRES_PATH, "r", encoding="utf-8") as g_file:
            genres_to_process = [line.rstrip("\n") for line in g_file]
    artists_to_process = artist_info_collector.get_top_artists_for_genre(
        genres=genres_to_process,
        result_path=ARTISTS_FILE,
        remaining_path=REMAINING_GENRES_PATH,
        save_on=100,
    )

    if os.path.isfile(REMAINING_ARTISTS):
        with open(REMAINING_ARTISTS, "r", encoding="utf-8") as rg_file:
            artists_to_process = json.load(rg_file)
            rg_file.close()
        album_data = pd.read_json(
            ALBUM_DATA_PATH, orient="records", lines=True
        )
        artist_info_collector.cover_frame = album_data

    album_data = pd.read_json(ALBUM_DATA_PATH, orient="records", lines=True)
    artist_info_collector.cover_frame = album_data
    artist_info_collector.collect_album_cover(
        target_dir="./data/covers300", size=300
    )
    artist_info_collector.collect_album_cover(
        target_dir="./data/covers64", size=64
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
