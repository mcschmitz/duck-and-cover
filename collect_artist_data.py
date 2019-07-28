import json
import os

import spotipy
import spotipy.util as util
from tqdm import tqdm

from spotify_client import get_client_id, get_client_secret


class SpotifyInfoCollector:

    def __init__(self, spotify_id: str, spotify_secret: str, artists: dict = None):
        """
        Collector that gathers information about artists via the spotify API
        Args:
            spotify_id: Spotify API client ID
            spotify_secret: Spotify API Client secret
            artists: Dictionary with existing artist information. None by default to create a new information
                dictionary
        """
        self.client_id = spotify_id
        self.client_secret = spotify_secret
        self.token = None
        self.generate_new_token()
        self.generate_spotify_session()
        self.artists = artists if artists is not None else {}
        self.remaining_genres = []

    def generate_new_token(self):
        """
        Generates a new token for the Spotify API from the class object client id and client secret
        """
        self.token = util.oauth2.SpotifyClientCredentials(self.client_id, self.client_secret).get_access_token()

    def generate_spotify_session(self):
        """
        Generates a new Spotify session object based on the provided token and returns it
        """
        return spotipy.Spotify(self.token)

    def get_artist_genre(self, artist, fall_back_genre):
        """
        Gets the requested information about an artits. Currently only the genre is extracted
        Args:
            artist: Artists result of the Spotify API
            fall_back_genre: Fallback Genre if there is no genre information in the given artist result

        Returns:

        """
        artist_id = artist["id"]
        ra_name = artist["name"]
        ra_genres = artist["genres"] if artist["genres"] else [fall_back_genre]
        self.artists[artist_id] = {"name": ra_name, "genres": ra_genres}

    def get_top_genre_artists(self, genres, save_on: int = 100, result_path: str = None, remaining_path: str = None):
        """
        Collects top artists for a list of genres and their recommended artists

        Args:
            genres: List of genres
            save_on: Save result after 'save_on' iterations. Ignored if path is None
            result_path: Where to save the result
            remaining_path: Where to save the list of the remaining genres

        """
        self.remaining_genres = genres
        spotify_session = self.generate_spotify_session()

        for idx, genre in tqdm(enumerate(genres)):
            try:
                result = spotify_session.search('genre:"{}"'.format(genre), type="artist", limit=50)
            except Exception as ex:
                print(ex)
                print("Generating new token")
                self.generate_new_token()
                spotify_session = self.generate_spotify_session()
                result = spotify_session.search('genre:"{}"'.format(genre), type="artist", limit=50)

            result_artists = result["artists"]["items"]
            for result_artist in result_artists:
                artist_id = result_artist["id"]
                if artist_id in self.artists:
                    continue
                self.get_artist_genre(result_artist, genre)
                try:
                    related_artists = spotify_session.artist_related_artists(artist_id)["artists"]
                except:
                    # @ TODO ExceptionType
                    print("Generating new token")
                    self.generate_new_token()
                    spotify_session = self.generate_spotify_session()
                    related_artists = spotify_session.artist_related_artists(artist_id)["artists"]
                for rel_a in related_artists:
                    self.get_artist_genre(rel_a, genre)

            if idx % 100 == 0:
                spotify_session = self.generate_spotify_session()

            if idx % save_on == 0 and result_path is not None:
                with open(result_path, "w", encoding="utf-8") as result_file:
                    json.dump(self.artists, result_file)

                self.remaining_genres = genres[idx:]
                with open(remaining_path, "w", encoding="utf-8") as remaining_file:
                    for remaining_genre in self.remaining_genres:
                        remaining_file.write("%s\n" % remaining_genre)

        with open(result_path, "w", encoding="utf-8") as remaining_file:
            json.dump(self.artists, remaining_file)
            self.remaining_genres = []


ARTIST_DATA_PATH = "data/artist_data/artist_data.json"
GENRES_PATH = "data/genres.txt"
REMAINING_GENRES_PATH = "tmp/remaining_genres.txt"

if __name__ == "__main__":
    CLIENT_ID = get_client_id()
    CLIENT_SECRET = get_client_secret()

    if os.path.isfile(ARTIST_DATA_PATH):
        with open("data/artist_data/artist_data.json", "r", encoding="utf-8") as file:
            EXISTING_ARTISTS = json.load(file)
    else:
        EXISTING_ARTISTS = None

    if os.path.isfile(REMAINING_GENRES_PATH):
        GENRES_TO_PROCESS = [line.rstrip("\n") for line in open(REMAINING_GENRES_PATH)]
    else:
        GENRES_TO_PROCESS = [line.rstrip("\n") for line in open(GENRES_PATH)]

    ARTIST_INFO_COLLECTOR = SpotifyInfoCollector(spotify_id=CLIENT_ID, spotify_secret=CLIENT_SECRET,
                                                 artists=EXISTING_ARTISTS)
    ARTIST_INFO_COLLECTOR.get_top_genre_artists(GENRES_TO_PROCESS, 100, ARTIST_DATA_PATH, REMAINING_GENRES_PATH)
