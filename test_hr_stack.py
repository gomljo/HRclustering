from HRclustering.HR_clustering_stack import HR_clustering
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from HRclustering.make_data import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import time
import matplotlib.pyplot as plt

scaler = MinMaxScaler()

dataset = load_iris()

x_data = dataset['data']
y_data = dataset['target']
# label_names_ = dataset['target_names']

# x_data = x
# y_data = y.ravel()
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
    # DATA_NAME = types[types_num] + '_' + str(n_clusters) + '_' + str(n_features) + '_' + str(samples) # types_n_clusters_n_features
    # CONDITION = r'ACC_eps_interval_{}'.format(itv)
    # CONDITION2 = r'ARI_eps_interval_{}'.format(itv)
    # DIR_PATH = os.path.join(DIR_PATH, DATA_NAME)
    # DIR_PATH_ACC = os.path.join(DIR_PATH, CONDITION)
    # DIR_PATH_ARI = os.path.join(DIR_PATH, CONDITION2)

    # if not os.path.exists(DIR_PATH_ACC):
    #     os.makedirs(DIR_PATH_ACC)

    # if not os.path.exists(DIR_PATH_ARI):
    #     os.makedirs(DIR_PATH_ARI)

    for eps in epsilon:

        hrdbscan_ARI = []
        dbscan_ARI = []
        hdbscan_ARI = []

        for min_sam in min_samples:
        # print(type(min_sam))
        # min_sam = 3
            model1 = DBSCAN(eps=eps, min_samples=min_sam)
            model1.fit(x_data)
            dbscan_ARI.append(adjusted_rand_score(y_data, model1.labels_))

            model2 = HR_clustering(X=x_data, tau=eps)
            start = time.time()
            model2.fit_predict(min_samples=min_sam)
            end = time.time()
            duration = end - start
            # print(model2.labels_)
            print('HRDBSCAN\'s ARI: ', adjusted_rand_score(y_data, model2.labels_))
            hrdbscan_ARI.append(adjusted_rand_score(y_data, model2.labels_))
