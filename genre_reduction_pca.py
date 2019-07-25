import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA

GENRE_DF_PATH = "data/GenreEncoder/genre_data.h5"
genre_df = pd.DataFrame(pd.read_hdf(GENRE_DF_PATH, "df"))

pca = PCA()
pca.fit(genre_df)

ax = sns.lineplot(x=range(pca.n_components_), y=np.cumsum(pca.explained_variance_ratio_))
ax.axhline(.75, ls="--", c="black")
ax.axhline(.9, ls="--", c="black")
ax.axhline(.95, ls="--", c="black")
ax.set_title("Cumulative explained Variance")
ax.set_xlabel("No. of features")
ax.set_ylabel("Cumulative explained Variance")
plt.savefig("GenreEncoder/plots/PCA_cumulative_explained_var.png")
plt.close()

ax = sns.barplot(x=["c{}".format(i) for i in range(pca.n_components_)], y=pca.explained_variance_ratio_)
ax.set_title("Explained Variance per feature")
ax.set_xlabel("Features")
ax.set_ylabel("Explained Variance")
plt.savefig("GenreEncoder/plots/PCA_explained_var.png")
plt.close()

kpca = KernelPCA(n_jobs=-1)
kpca.fit(genre_df)