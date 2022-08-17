from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt


center = [(2, 2),(10,5)]

X, y = make_blobs(n_samples=50, centers=center, n_features=2, random_state=42)
class_1_idx = list(np.where(y == 0))
class_2_idx = list(np.where(y == 1))
print(len(class_1_idx))
print(X[class_1_idx].shape)
print(X[class_2_idx])

plt.scatter(X[class_1_idx][:,0], X[class_1_idx][:,1], c='red', label='cluster1')
plt.scatter(X[class_2_idx][:,0], X[class_2_idx][:,1], c='green', label='cluster2')
plt.xlabel('X1', fontsize=20)
plt.ylabel('X2', fontsize=20)
plt.title('Two cluster in dataset', fontsize=20)
plt.legend(fontsize=15)
plt.show()
