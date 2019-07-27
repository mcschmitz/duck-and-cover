import json

import spotipy
import spotipy.util as util
from tqdm import tqdm

from spotify_client import get_client_id, get_client_secret


class SpotifyInfoCollector:

    def __init__(self, token, artists: dict = None):
        """
        Collector that gathers information about artists via the spotify API
        Args:
            token: Spotify API token
            artists: Dictionary with existing artist information. None by default to create a new information
                dictionary
        """
        self.token = token
        self.artists = artists if artists is not None else {}
        self.remaining_genres = []

    def generate_new_token(self, client_id, client_secret):
        """
        @ Todo
        Args:
            client_id:
            client_secret:

        Returns:

        """
        self.token = util.oauth2.SpotifyClientCredentials(client_id, client_secret).get_access_token()

    def generate_spotify_session(self):
        """
        Generates a new Spotify session object based on the provided token and returns it
        Returns:

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
        ra_genres = artist["genres"] if len(artist["genres"]) else [fall_back_genre]
        self.artists[artist_id] = {"name": ra_name, "genres": ra_genres}

    def get_top_genre_artists(self, genres, save_on: int = 100, path: str = None):
        """
        Collects top artists for a list of genres and their recommended artists

        Args:
            genres: List of genres
            save_on: Save result after 'save_on' iterations. Ignored if path is None
            path: Where to save the result

        """
        self.remaining_genres = genres
        spotify_session = self.generate_spotify_session()

        for idx, genre in tqdm(enumerate(genres)):
            result = spotify_session.search('genre:"{}"'.format(genre), type="artist", limit=50)
            result_artists = result["artists"]["items"]
            for ra in result_artists:
                self.get_artist_genre(ra, genre)
                artist_id = ra["id"]
                related_artists = spotify_session.artist_related_artists(artist_id)["artists"]
                for rel_a in related_artists:
                    self.get_artist_genre(rel_a, genre)

            if idx % 100 == 0:
                spotify_session = self.generate_spotify_session()

            if idx % save_on == 0 and path is not None:
                with open(path, "w", encoding="utf-8") as file:
                    json.dump(self.artists, file)
                    self.remaining_genres = genres[idx:]

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.artists, file)
            self.remaining_genres = genres[idx:]


client_id = get_client_id()
client_secret = get_client_secret()

token = util.oauth2.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret).get_access_token()
genres = [line.rstrip("\n") for line in open("data/genres.txt")]

artist_info_collector = SpotifyInfoCollector(token)
artist_info_collector.get_top_genre_artists(genres, 100, "data/artist_data/artist_data.json")

# rerun remaining genres
while len(artist_info_collector.remaining_genres) > 0:
    artist_info_collector.generate_new_token(client_id, client_secret)
    artist_info_collector.generate_spotify_session()
    artist_info_collector.get_top_genre_artists(artist_info_collector.remaining_genres, 100,
                                                "data/artist_data/artist_data.json")
