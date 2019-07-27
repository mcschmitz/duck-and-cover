import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def score(correlation_matrix):
    """
    Function to assign a score to an ordered covariance matrix.
    High correlations within a cluster improve the score.
    High correlations between clusters decease the score.
    """
    score = 0
    for cluster in range(n_clusters):
        inside_cluster = np.arange(cluster_size) + cluster * cluster_size
        outside_cluster = np.setdiff1d(range(n_variables), inside_cluster)

        # Belonging to the same cluster
        score += np.sum(correlation_matrix[inside_cluster, :][:, inside_cluster])

        # Belonging to different clusters
        score -= np.sum(correlation_matrix[inside_cluster, :][:, outside_cluster])
        score -= np.sum(correlation_matrix[outside_cluster, :][:, inside_cluster])

    return score


def swap_rows(C, var1, var2):
    '''
    Function to swap two rows in a covariance matrix,
    updating the appropriate columns as well.
    '''
    D = C.copy()
    D[var2, :] = C[var1, :]
    D[var1, :] = C[var2, :]

    E = D.copy()
    E[:, var2] = D[:, var1]
    E[:, var1] = D[:, var2]

    return E


# General Parameters
GENRE_CORR_PATH = "data/GenreEncoder/genre_correlation.h5"
genre_corr = pd.read_hdf(GENRE_CORR_PATH, "df")

n_variables = genre_corr.shape[0]
n_clusters = 50
cluster_size = n_variables // n_clusters

# Assign each variable to a cluster
belongs_to_cluster = np.repeat(range(n_clusters), cluster_size)
np.random.shuffle(belongs_to_cluster)

initial_corr = genre_corr.values
initial_score = score(initial_corr)
initial_ordering = np.arange(n_variables)

plt.figure()
plt.imshow(initial_corr, interpolation='nearest')
plt.title('Initial Correlation Matrix')
plt.savefig("GenreEncoder/plots/CorrelationBlocks_initial_correlation.png")
plt.close()
print('Initial ordering:', initial_ordering)
print('Initial covariance matrix score:', initial_score)

current_corr = initial_corr
current_ordering = initial_ordering
current_score = initial_score

max_iter = 1000
for i in range(max_iter):
    # Find the best row swap to make
    best_C = current_corr
    best_ordering = current_ordering
    best_score = current_score
    for row1, row2 in itertools.product(range(n_variables), range(n_variables)):
        if row1 == row2:
            continue
        option_ordering = best_ordering.copy()
        option_ordering[row1] = best_ordering[row2]
        option_ordering[row2] = best_ordering[row1]
        option_C = swap_rows(best_C, row1, row2)
        option_score = score(option_C)

        if option_score > best_score:
            best_C = option_C
            best_ordering = option_ordering
            best_score = option_score

    if best_score > current_score:
        # Perform the best row swap
        current_corr = best_C
        current_ordering = best_ordering
        current_score = best_score
    else:
        # No row swap found that improves the solution, we're done
        break

# Output the result
plt.figure()
plt.imshow(current_corr, interpolation='nearest')
plt.title('Best C')
print('Best ordering:', current_ordering)
print('Best score:', current_score)
print('Cluster     [variables assigned to this cluster]')
print('------------------------------------------------')
for cluster in range(n_clusters):
    print('Cluster %02d  %s' % (cluster + 1, current_ordering[cluster * cluster_size:(cluster + 1) * cluster_size]))
