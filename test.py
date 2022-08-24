from HRclustering.HR_clustering import HR_clustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from HRclustering.make_data import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

import matplotlib.pyplot as plt

scaler = MinMaxScaler()

# dataset = load_iris()
types = ['classification', 'moon', 'blobs', 'circles']
n_clusters = 3
n_features = 2
samples = 500
n_clusters_per_class = 1
types_num = 3
idx, x, y = generate(type_=types[types_num], n_cluster=n_clusters, n_features=n_features, n_samples=samples,
                     n_clusters_per_class=n_clusters_per_class)
plot_data(x, y)
# x_data = dataset['data']
# y_data = dataset['target']
# label_names_ = dataset['target_names']

x_data = x
y_data = y.ravel()
# label_names_ = np.unique(y)

x_data = scaler.fit_transform(x_data)

interval = [10]

label_mask = []

for label in np.unique(y_data):
    mask = list(np.where(y_data == label))
    label_mask.append(mask)

for itv in interval:
    epsilon = np.linspace(0.001, 1.0, itv)

    min_samples = np.arange(2, len(x_data), 1)

    DIR_PATH = r'C:\Users\YoungHo\Documents\Cloud\ML_Project\HRclustering\data'
    DATA_NAME = types[types_num] + '_' + str(n_clusters) + '_' + str(n_features) + '_' + str(samples)# types_n_clusters_n_features
    # CONDITION = r'ACC_eps_interval_{}'.format(itv)
    CONDITION2 = r'ARI_eps_interval_{}'.format(itv)
    DIR_PATH = os.path.join(DIR_PATH, DATA_NAME)
    # DIR_PATH_ACC = os.path.join(DIR_PATH, CONDITION)
    DIR_PATH_ARI = os.path.join(DIR_PATH, CONDITION2)

    # if not os.path.exists(DIR_PATH_ACC):
    #     os.makedirs(DIR_PATH_ACC)

    if not os.path.exists(DIR_PATH_ARI):
        os.makedirs(DIR_PATH_ARI)

    for eps in epsilon:

        hrdbscan_acc = []
        dbscan_acc = []

        hrdbscan_ARI = []
        dbscan_ARI = []

        for min_sam in min_samples:

            model1 = DBSCAN(eps=eps, min_samples=min_sam)
            model1.fit(x_data)
            dbscan_ARI.append(adjusted_rand_score(y_data, model1.labels_))

            model2 = HR_clustering(X=x_data, tau=eps)
            model2.fit_predict(min_samples=min_sam)
            hrdbscan_ARI.append(adjusted_rand_score(y_data, model2.labels_))

            # for mask in label_mask:
            #
            #     unique_val_1, cnt_1 = np.unique(model1.labels_[mask], return_counts=True)
            #     model1.labels_[mask] = unique_val_1[np.argmax(cnt_1, axis=0)]
            #
            #     unique_val_2, cnt_2 = np.unique(model2.labels_[mask], return_counts=True)
            #     model2.labels_[mask] = unique_val_2[np.argmax(cnt_2, axis=0)]

            # dbscan_acc.append(accuracy_score(y_data, model1.labels_))
            # hrdbscan_acc.append(accuracy_score(y_data, model2.labels_))

        # FILE_NAME = 'eps_{:.3f}_min_sample_{}_{}.png'.format(eps, np.min(min_samples), np.max(min_samples))
        # PATH_ACC = os.path.join(DIR_PATH_ACC, FILE_NAME)
        # plt.plot(np.arange(len(min_samples)), dbscan_acc, c='red', label='DBSCAN')
        # plt.plot(np.arange(len(min_samples)), hrdbscan_acc, c='blue', label='HR-DBSCAN')
        # plt.ylabel('Accuracy', fontsize=15)
        # plt.ylim(0.0, 1.1)
        # plt.xlabel('min samples', fontsize=15)
        # plt.title('test on epsilon {}'.format(eps), fontsize=20)
        # plt.grid()
        # plt.legend()
        # plt.savefig(PATH_ACC)
        # plt.close()

        FILE_NAME = 'eps_{:.3f}_min_sample_{}_{}.png'.format(eps, np.min(min_samples), np.max(min_samples))
        PATH_ARI = os.path.join(DIR_PATH_ARI, FILE_NAME)
        plt.plot(np.arange(len(min_samples)), dbscan_ARI, c='red', label='DBSCAN')
        plt.plot(np.arange(len(min_samples)), hrdbscan_ARI, c='blue', label='HR-DBSCAN')
        plt.ylabel('Adjusted Rand Index', fontsize=15)
        plt.ylim(0.0, 1.0)
        plt.xlabel('min samples', fontsize=15)
        plt.title('test on epsilon {}'.format(eps), fontsize=20)
        plt.grid()
        plt.legend()
        plt.savefig(PATH_ARI)
        plt.close()
