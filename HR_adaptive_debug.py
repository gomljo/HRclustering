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
    # X, y = Data_loader_Toy('iris', is_scale=True)
    X, y = Data_loader_IoT('./data/entropy6_v1.csv', is_scale=True, class_per_sample=10000)
    print(X)
    sleep(0.1)
    # print(X.shape[0])
    best = 0
    best_min = 0
    dura = 0
    cls = 0
    for i in tqdm(range(2, 100)):
        model1 = Adaptive_hyper_rectangle(min_pts=i)
        start = time.time()
        label = model1.fit_predict(X=X)
        # print('model1\'s num. of cluster: ', len(model1.clusters))
        # print(len(model1.clusters))
        # model1.cluster_info()
        end = time.time()
        # model1.visualize()
        duration = end - start

        # print('model1\'s duration: ', duration)
        if best < adjusted_rand_score(y.ravel(), label):
            best = adjusted_rand_score(y.ravel(), label)
            best_min = i
            dura = duration
            cls = len(model1.clusters)
        # print('model1\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), label))
        # print('model1\'s rand index: ', rand_score(y.ravel(), label))
    print('my model')
    print('best min pts:{}, best time: {}, num. of cluster: {}, best ARI: {}'.format(best_min, dura, cls, best))
