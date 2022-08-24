from sklearn.metrics.cluster import rand_score, adjusted_rand_score
from scipy.special import comb

import numpy as np
ground_truth = np.array([
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1]
])

clustering_result =\
    np.array(
[
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1]
]
)

incident_mat = 1 * (ground_truth == clustering_result)
print(incident_mat)

np.fill_diagonal(incident_mat, 0)
sum = np.sum(incident_mat) / 2

rand_index = sum / comb(incident_mat.shape[0], 2)
print(rand_index)
