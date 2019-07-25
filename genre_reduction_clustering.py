import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN

plt.ioff()

# General Parameters
GENRE_CORR_PATH = "data/GenreEncoder/genre_correlation.h5"
GENRE_DATA_PATH = "data/GenreEncoder/genre_data.h5"

# Hierarchical clustering
genre_corr = pd.DataFrame(pd.read_hdf(GENRE_CORR_PATH, "df"))
l = linkage(genre_corr, method="centroid")

plt.figure(figsize=(25, 10))
plt.title("Dendogram for Correlation Clustering")
d = dendrogram(l, p=50, truncate_mode="lastp")
plt.savefig("GenreEncoder/plots/correlation_clustering_dendogram_.png")

# DBSCAN
genre_data = pd.DataFrame(pd.read_hdf(GENRE_DATA_PATH, "df"))
dbscan = DBSCAN(min_samples=10, metric="jaccard", n_jobs=-1, eps=.5)
labels = dbscan.fit_predict(genre_data)
np.unique(labels, return_counts=True)
