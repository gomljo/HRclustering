from HRclustering.HR_adaptive import *
from HRclustering.data_loader import *
from sklearn.metrics import adjusted_rand_score, rand_score

from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from HRclustering.HR_clustering import *

import time
from time import sleep

if __name__ == '__main__':

    # X, y = Data_loader_synthetic('classification', n_cluster=10, n_features=20, n_samples=10000, is_scale=True)
    # X, y = Data_loader_synthetic('moon', n_cluster=2, n_features=2, n_samples=500, is_scale=True)
    # X, y = Data_loader_synthetic('blobs', n_cluster=3, n_features=2, n_samples=60, is_scale=True)
    # X, y = Data_loader_synthetic('circles', n_cluster=2, n_features=2, n_samples=60, is_scale=True)
    # X, y = Data_loader_Toy('breast_cancer', is_scale=True)

    X, y = Data_loader_IoT('./data/entropy6_v1.csv', is_scale=True, class_per_sample=10000)

    sleep(0.1)
    # print(X.shape[0])
    best = 0
    best_min = 0
    best_eps = 0.0
    epsilon = np.arange(0.01, 1, 0.01)
    dura = 0.0
    cls = 0
    # for eps in epsilon:
    for i in tqdm(range(2, 100)):

        for eps in epsilon:
            model2 = HDBSCAN(cluster_selection_epsilon=float(eps), min_cluster_size=int(i))
            start = time.time()
            model2.fit_predict(X)
            end = time.time()
            duration = end - start
            if best < adjusted_rand_score(y.ravel(), model2.labels_):
                best = adjusted_rand_score(y.ravel(), model2.labels_)
                best_min = i
                best_eps = eps
                dura = duration
                cls = len(np.unique(model2.labels_))
            # print('model2\'s num. of cluster: ', len(np.unique(model2.labels_)))
            # print('model2\'s duration: ', duration)
            # print('model2\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), model2.labels_))
            # print('model2\'s rand index: ', rand_score(y.ravel(), model2.labels_))
    print('HDBSCAN')
    print('best min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}'.format(best_min, best_eps,
                                                                                                   dura, cls, best))
    print()
    sleep(0.1)

    best = 0
    best_min = 0
    best_eps = 0.0
    epsilon = np.arange(0.01, 1, 0.01)
    dura = 0.0
    cls = 0
    # for eps in epsilon:
    for i in tqdm(range(2, 100)):
        # best = 0
        # best_min = 0
        # best_eps = 0.0
        # dura = 0.0
        # cls = 0
        # sleep(0.001)
        for eps in epsilon:
            model3 = DBSCAN(eps=eps, min_samples=i)
            start = time.time()
            model3.fit_predict(X)
            end = time.time()
            duration = end - start
            if best < adjusted_rand_score(y.ravel(), model3.labels_):
                best = adjusted_rand_score(y.ravel(), model3.labels_)
                best_min = i
                best_eps = eps
                dura = duration
                cls = len(np.unique(model3.labels_))
            # print('model3\'s num. of cluster: ', len(np.unique(model3.labels_)))
            # print('model3\'s duration: ', duration)
            # print('model3\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), model3.labels_))
            # print('model3\'s rand index: ', rand_score(y.ravel(), model3.labels_))
    print('DBSCAN')
    print('best min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}'.format(best_min, best_eps,
                                                                                                   dura,
                                                                                   cls, best))
    sleep(0.1)

