import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from GenreEncoding.build_dataframe import ARTIST_GENRE_PATH

artist_genre = pd.DataFrame(pd.read_hdf(ARTIST_GENRE_PATH, "df"))

# Detect sole genres
sole_genres = []
for genre in tqdm(artist_genre.columns):
    sub_data = artist_genre[artist_genre[genre] == 1]
    if np.sum(sub_data.values) == len(sub_data):
        sole_genres.append(genre)

freq = artist_genre.sum(axis=0)
freq = freq.sort_values(ascending=False)

# Get Genre Frequency
ax = sns.kdeplot(data=freq)
ax.set_title("Density of Genre Frequency")
ax.set_xlabel("Frequency")
ax.set_ylabel("Density")
plt.savefig("GenreEncoding/plots/genre_dens.png")
plt.close()

# Get Cumulative Frequency
ax = sns.lineplot(x=range(len(freq)), y=freq.cumsum() / freq.sum())
ax.set_title("Relative Cumulative Ratio of Genres")
ax.set_xlabel("# Genres")
ax.set_ylabel("Relative Cumulative Ratio")
plt.savefig("GenreEncoding/plots/genre_cumsum.png")
plt.close()
