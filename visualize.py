from sklearn.datasets import load_breast_cancer
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

dataset = load_breast_cancer()
x = dataset['data']
y = dataset['target']

mask = np.where(y==0)
print(mask)
mask_2 = np.where(y==1)
dim = x.shape[1]

comb = np.arange(dim)

comb = list(combinations(comb, 2))

for idx, c in enumerate(comb):
    # plt.subplot(5, 5, (idx+1)//25)
    plt.scatter(x[mask, c[0]], x[mask, c[1]], c='red')
    plt.scatter(x[mask_2, c[0]], x[mask_2, c[1]], c='black')
    plt.show()
