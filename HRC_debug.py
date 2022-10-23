from HRclustering.data_loader import *
from sklearn.metrics import adjusted_rand_score, rand_score

from HRclustering.HR_clustering import *

import time
from time import sleep

if __name__ == '__main__':

    # X, y = Data_loader_synthetic('classification', n_cluster=10, n_features=20, n_samples=10000, is_scale=True)
    # X, y = Data_loader_synthetic('moon', n_cluster=2, n_features=2, n_samples=500, is_scale=True)
    # X, y = Data_loader_synthetic('blobs', n_cluster=3, n_features=2, n_samples=60, is_scale=True)
    # X, y = Data_loader_synthetic('circles', n_cluster=2, n_features=2, n_samples=60, is_scale=True)
    # X, y = Data_loader_Toy('iris', is_scale=True)
    # print(X)
    X, y = Data_loader_IoT('./data/entropy6_v1.csv', is_scale=True, class_per_sample=10000)

    best = 0
    best_min = 0
    best_eps = 0.0
    epsilon = np.arange(0.01, 0.5, 0.01)
    dura = 0.0
    # for eps in epsilon:
    for i in tqdm(range(2, 50)):
        # print('min_sample: {}'.format(i))
        # sleep(0.1)
        for eps in (epsilon):
            model4 = HR_clustering(X=X, tau=eps)
            start = time.time()
            model4.fit_predict(min_samples=i)
            end = time.time()
            duration = end - start

            if best < adjusted_rand_score(y.ravel(), model4.labels_):
                best = adjusted_rand_score(y.ravel(), model4.labels_)
                best_min = i
                best_eps = eps
                dura = duration
                cls = len(np.unique(model4.labels_))

            # print('model4\'s num. of cluster: ', len(np.unique(model4.labels_)))
            # print('model4\'s duration: ', duration)
            # print('model4\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), model4.labels_))
            # print('model4\'s rand index: ', rand_score(y.ravel(), model4.labels_))
    print('HR-DBSCAN')
    print(
        'best min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}'.format(best_min, best_eps,
                                                                                                 dura, cls, best))