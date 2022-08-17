from HR_TREE.HRTREE import *
from sklearn.datasets import load_iris
import seaborn as sns
data = load_iris()

# print(data)

X = data['data']
y = data['target']
# iris = sns.load_dataset("iris")
# sns.scatterplot(x='petal_length', y='petal_width', data=iris, hue='species')
# plt.show()
hrtree = HRTREE(X, tau=0.3)
hrtree.fit()
a = np.array([
 [5,  3.6, 1.4, 0.2],
 [5.1, 3.4, 1.5, 0.2],
 [5.,  3.5, 1.3, 0.3],
 [5.,  3.4, 1.5, 0.2],
 [5.1, 3.5, 1.4, 0.3]
     ])

b = np.array([
 [5.1, 3.5, 1.4, 0.2],
 [5.1, 3.5, 1.4, 0.3],
 [5.,  3.5, 1.3, 0.3],
 [4.9, 3.6, 1.4, 0.1],
     ])

# plt.scatter(a[:,0], a[:,1], c='red')
# plt.scatter(b[:,0], b[:,1], c='blue')
# plt.show()
# print(np.mean(a))
# print(np.var(a))


def KLD(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

