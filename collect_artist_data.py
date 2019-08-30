import json
import os
import shutil

import pandas as pd
import spotipy
import spotipy.util as util
from tqdm import tqdm

from credentials import get_client_id, get_client_secret


class SpotifyInfoCollector:

    def __init__(self, spotify_token: str, spotify_id: str, spotify_secret: str, cover_frame: pd.DataFrame = None,
                 artist_genres_map: dict = None):
        """
        Collector that gathers information about artists via the spotify API
        Args:
            spotify_token: Spotify API Token. Automatically generated if not provided
            spotify_id: Spotify API client ID
            spotify_secret: Spotify API Client secret
            artist_genres_map: Already collected artist IDs and genres
            cover_frame: Dataframe of already collected album information
        """
        self.client_id = spotify_id
        self.client_secret = spotify_secret
        self.artist_genres = artist_genres_map if artist_genres_map else dict()
        self.token = spotify_token if spotify_token else self.create_new_token()
        self.create_spotify_session()
        self.remaining_genres = list()
        self.remaining_artists = list()
        self.spotify_session = self.create_spotify_session()
        if cover_frame is not None:
            self.cover_frame = cover_frame
        else:
            self.cover_frame = pd.DataFrame([], columns=["artist_id", "artist_name", "artist_genre",
                                                         "album_id", "album_name", "album_release", "album_cover_url"])

    def create_new_token(self):
        """
        Generates a new token for the Spotify API from the class object client id and client secret
        """
        return util.oauth2.SpotifyClientCredentials(self.client_id, self.client_secret).get_access_token()

    def create_spotify_session(self):
        """
        Generates a new Spotify session object based on the provided token and returns it
        """
        return spotipy.Spotify(self.token)

    def get_genre_artists(self, genres, save_on: int = 100, result_path: str = None, remaining_path: str = None):
        """
        Collects top artists for a list of genres and their recommended artists and saves their corresponding genres
        to the result path

        Args:
            genres: List of genres
            save_on: Save result after 'save_on' iterations. Ignored if path is None
            result_path: Where to save the result
            remaining_path: Where to save the list of the remaining genres

        Returns:
            Dict of artist IDs and their genres
        """
        self.remaining_genres = genres

        for idx, genre in tqdm(enumerate(genres)):
            try:
                result = self.spotify_session.search('genre:"{}"'.format(genre), type="artist", limit=50)
            except spotipy.client.SpotifyException:
                self.token = self.create_new_token()
                self.spotify_session = self.create_spotify_session()
                result = self.spotify_session.search('genre:"{}"'.format(genre), type="artist", limit=50)

            self.get_artist_ids_from_search(result, genre)

            if idx % 100 == 0:
                self.token = self.create_new_token()
                self.spotify_session = self.create_spotify_session()

            if idx % save_on == 0 and result_path is not None:
                self.write_to_json(self.artist_genres, result_path)
                self.remaining_genres = genres[idx:]
                self.write_to_txt(self.remaining_genres, remaining_path)

        self.write_to_json(self.artist_genres, result_path)
        os.remove(remaining_path)
        return self.artist_genres

    @staticmethod
    def write_to_json(obj, path):
        """
        Writes a given object as a JSON file to the path

        Args:
            obj: The object to save
            path: The target path
        """
        with open(path, "w", encoding="utf-8") as result_file:
            json.dump(obj, result_file, indent=4)
            result_file.close()

    @staticmethod
    def write_to_txt(obj, path):
        """
        Writes a given object as a text file to the path

        Args:
            obj: The object to save
            path: The target path
        """
        with open(path, "w", encoding="utf-8") as p:
            for f in obj:
                p.write("%s\n" % f)
            p.close()

    def get_artist_ids_from_search(self, result, fallback_genre: str = None):
        """
        Gets the artist IDs from a spotipy search result and writes an entry to the artist_genres dict that links the
        IDs with the provided genres. If the genre is missing the fallback genre is used

        Args:
            result: spotipy search result
            fallback_genre: Fallback genre. Used if no genre is provided for the given artist
        """
        artists = result["artists"]["items"]
        for artist in artists:
            artist_id = artist["id"]
            if artist_id not in self.artist_genres:
                self.artist_genres[artist_id] = {"genre": artist["genres"] if artist["genres"] else [fallback_genre]}
                try:
                    related_artists = self.spotify_session.artist_related_artists(artist_id)["artists"]
                except spotipy.client.SpotifyException:
                    self.token = self.create_new_token()
                    self.spotify_session = self.create_spotify_session()
                    related_artists = self.spotify_session.artist_related_artists(artist_id)["artists"]

                for related_artist in related_artists:
                    related_id = related_artist["id"]
                    if related_id not in self.artist_genres:
                        self.artist_genres[artist_id] = {
                            "genre": artist["genres"] if artist["genres"] else [fallback_genre]}

    def build_cover_data_frame(self, artists: dict = None, save_on: int = 100, result_path: str = None,
                               remaining_path: str = None):
        """
        Builds a dataframe containing all albums, their release date and an url to their cover for a given set of
        artists

        Args:
            artists: Dictionary of Spotify artist Ids and their gernes
            save_on: ave result after 'save_on' iterations. Ignored if path is None
            result_path: Where to save the result
            remaining_path: Where to save the list of the remaining genres
        """
        self.spotify_session = self.create_spotify_session()
        self.remaining_artists = artists.copy()

        for idx, artist_id in tqdm(enumerate(artists)):
            try:
                artist_albums = self.get_artist_album_data(artist_id, artists[artist_id]["genre"])
            except spotipy.client.SpotifyException:
                self.token = self.create_new_token()
                self.spotify_session = self.create_spotify_session()
                artist_albums = self.get_artist_album_data(artist_id, artists[artist_id]["genre"])

            for album in artist_albums:
                self.cover_frame = self.cover_frame.append(album, ignore_index=True)

            self.remaining_artists.pop(artist_id)

            if idx % 100 == 0:
                self.token = self.create_new_token()
                self.spotify_session = self.create_spotify_session()

            if idx % save_on == 0 and result_path is not None:
                if os.path.isfile(result_path):
                    source = result_path
                    name, ext = os.path.splitext(result_path)
                    name += "_copy"
                    target = name + ext
                    shutil.copyfile(source, target)
                self.cover_frame.to_json(result_path, lines=True, orient="records")
                self.write_to_json(self.remaining_artists, remaining_path)

        self.write_to_json(self.remaining_artists, remaining_path)
        self.cover_frame.to_json(result_path, lines=True, orient="records")
        os.remove(remaining_path)

    def get_artist_album_data(self, artist: str = None, genre: list = None):
        """
        Collects essential information about all  albums released by the given artist

        Args:
            artist: Spotify artist Id
            genre: Genre of the artist
        Returns:
            List of summarizing dictionaries of all albums released by an artist
        """

        albums = self.spotify_session.artist_albums(artist, album_type="album", limit=50)
        if albums:
            albums = albums["items"]
            result = []
            if len(albums) > 0:
                for album in albums:
                    if not album["images"] or album["type"] != "album":
                        continue
                    else:
                        images = album["images"]
                        cover_url = [i["url"] for i in images if i["height"] == 300]
                        if len(cover_url) > 0:
                            release_year = int(album["release_date"][:4])
                            album_summary = {"artist_id": artist, "artist_name": album["artists"][0]["name"],
                                             "artist_genre": genre,
                                             "album_id": album["id"], "album_name": album["name"],
                                             "album_release": release_year,
                                             "album_cover_url": cover_url[0]}
                            result.append(album_summary)
            return result
        else:
            return []


def get_artist_genres(spotify_info_collector):
    """
    Wrapper to return a dict that contains a set of artists for each given genre and their genres
    Args:
        spotify_info_collector: SpotifyInfoCollector instance

    Returns:
        Dict of artist IDs and their genres
    """
    if os.path.isfile(REMAINING_GENRES_PATH):
        genres_to_process = [line.rstrip("\n") for line in open(REMAINING_GENRES_PATH)]
        with open(ARTISTS_FILE, "r", encoding="utf-8") as file:
            artist_genres = json.load(file)
            file.close()
            spotify_info_collector = SpotifyInfoCollector(token, spotify_id=client_id, spotify_secret=client_secret,
                                                          artist_genres_map=artist_genres)
    else:
        genres_to_process = [line.rstrip("\n") for line in open(GENRES_PATH)]
    artist_genres = spotify_info_collector.get_genre_artists(genres_to_process, 100, ARTISTS_FILE,
                                                             REMAINING_GENRES_PATH)
    return artist_genres

ARTISTS_FILE = "data/artist_ids.json"
GENRES_PATH = "data/genres.txt"
REMAINING_GENRES_PATH = "tmp/remaining_genres.txt"
REMAINING_ARTISTS = "tmp/remaining_artists.json"
ALBUM_DATA_PATH = "data/album_data_frame.json"

if __name__ == "__main__":

    client_id = get_client_id()
    client_secret = get_client_secret()
    token = util.oauth2.SpotifyClientCredentials(client_id, client_secret).get_access_token()
    artist_info_collector = SpotifyInfoCollector(token, spotify_id=client_id, spotify_secret=client_secret)

    # artists_to_process = get_artist_genres(artist_info_collector)

    if os.path.isfile(REMAINING_ARTISTS):
        with open(REMAINING_ARTISTS, "r", encoding="utf-8") as file:
            artists_to_process = json.load(file)
            file.close()
        album_data = pd.read_json(ALBUM_DATA_PATH, orient="records", lines=True)
        artist_info_collector = SpotifyInfoCollector(token, spotify_id=client_id, spotify_secret=client_secret,
                                                     cover_frame=album_data)
    else:
        with open(ARTISTS_FILE, "r", encoding="utf-8") as file:
            artists_to_process = json.load(file)
            file.close()
    artist_info_collector.build_cover_data_frame(artists_to_process, 1000, ALBUM_DATA_PATH, REMAINING_ARTISTS)
