from HRclustering.HR_adaptive import *
from HRclustering.data_loader import *
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from HRclustering.HR_clustering import *
from datetime import datetime
import time
from time import sleep

if __name__ == '__main__':

    # X, y = Data_loader_synthetic('classification', n_cluster=10, n_features=20, n_samples=10000, is_scale=True)
    # X, y = Data_loader_synthetic('moon', n_cluster=2, n_features=2, n_samples=500, is_scale=True)
    # X, y = Data_loader_synthetic('blobs', n_cluster=3, n_features=2, n_samples=60, is_scale=True)
    # X, y = Data_loader_synthetic('circles', n_cluster=2, n_features=2, n_samples=60, is_scale=True)
    name, X, y = Data_loader_Toy('wine', is_scale=True)
    # name, X, y = Data_loader_IoT('./data/entropy6_v1.csv', is_scale=True, class_per_sample=10000)
    # print(X)
    text = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = text
    text += '\n' + 'Dataset Name: ' + name + '\n'
    # print(text)
    file_name += '_'+name+'.txt'

    # sleep(0.1)
    # print(X.shape[0])
    best = 0
    best_NMI = 0
    best_min = 0
    dura = 0
    cls = 0
    max_sam = 100
    for i in tqdm(range(2, max_sam)):
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
            best_NMI = normalized_mutual_info_score(y.ravel(), label)
            best_min = i
            dura = duration
            cls = len(model1.clusters)
        # print('model1\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), label))
        # print('model1\'s rand index: ', rand_score(y.ravel(), label))
    print('my model')
    print(
        'best min pts:{}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}'.format(best_min,
                                                                                                               dura,
                                                                                                               cls,
                                                                                                               best,
                                                                                                               best_NMI))
    text += 'HR-DBSCAN\nbest min pts:{}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}\n'.format(
        best_min,
        dura, cls, best, best_NMI)

    best = 0
    best_min = 0
    best_eps = 0.0
    epsilon = np.arange(0.01, 1, 0.01)
    dura = 0.0
    cls = 0
    # for eps in epsilon:
    for i in tqdm(range(2, max_sam)):
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
                best_NMI = normalized_mutual_info_score(y.ravel(), model3.labels_)
                best_min = i
                best_eps = eps
                dura = duration
                cls = len(np.unique(model3.labels_))
            # print('model3\'s num. of cluster: ', len(np.unique(model3.labels_)))
            # print('model3\'s duration: ', duration)
            # print('model3\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), model3.labels_))
            # print('model3\'s rand index: ', rand_score(y.ravel(), model3.labels_))
    print('DBSCAN')
    print(
        'best min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}'.format(best_min,
                                                                                                               best_eps,
                                                                                                               dura,
                                                                                                               cls,
                                                                                                               best,
                                                                                                               best_NMI))
    text += 'DBSCAN\nbest min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}\n'.format(
        best_min, best_eps,
        dura, cls, best, best_NMI)

    sleep(0.1)

    best = 0
    best_min = 0
    best_eps = 0.0
    epsilon = np.arange(0.01, 1, 0.01)
    dura = 0.0
    cls = 0
    # for eps in epsilon:
    for i in tqdm(range(2, max_sam)):

        for eps in epsilon:
            model2 = HDBSCAN(cluster_selection_epsilon=float(eps), min_cluster_size=int(i))
            start = time.time()
            model2.fit_predict(X)
            end = time.time()
            duration = end - start
            if best < adjusted_rand_score(y.ravel(), model2.labels_):
                best = adjusted_rand_score(y.ravel(), model2.labels_)
                best_NMI = normalized_mutual_info_score(y.ravel(), model2.labels_)
                best_min = i
                best_eps = eps
                dura = duration
                cls = len(np.unique(model2.labels_))
            # print('model2\'s num. of cluster: ', len(np.unique(model2.labels_)))
            # print('model2\'s duration: ', duration)
            # print('model2\'s adjusted rand index: ', adjusted_rand_score(y.ravel(), model2.labels_))
            # print('model2\'s rand index: ', rand_score(y.ravel(), model2.labels_))
    print('HDBSCAN')
    print(
        'best min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}'.format(best_min,
                                                                                                               best_eps,
                                                                                                               dura,
                                                                                                               cls,
                                                                                                               best,
                                                                                                               best_NMI))
    text += 'HDBSCAN\nbest min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}\n'.format(
        best_min, best_eps,
        dura, cls, best, best_NMI)
    sleep(0.1)

    best = 0
    best_min = 0
    best_eps = 0.0
    epsilon = np.arange(0.01, 0.5, 0.01)
    dura = 0.0
    # for eps in epsilon:
    for i in tqdm(range(2, max_sam)):
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
                best_NMI = normalized_mutual_info_score(y.ravel(), model4.labels_)
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
        'best min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}'.format(best_min, best_eps,
                                                                                                 dura, cls, best, best_NMI))
    text += 'HR-DBSCAN\nbest min pts:{}, best eps: {}, best time: {}, num. of cluster: {}, best ARI: {}, best NMI: {}\n'.format(best_min, best_eps,
                                                                                                 dura, cls, best, best_NMI)
    txt = open('./result/' + file_name, 'w')
    txt.write(text)
    txt.close()