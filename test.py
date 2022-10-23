from HRclustering.HR_clustering import HR_clustering
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from HRclustering.make_data import *
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score, rand_score
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import os
import time
import matplotlib.pyplot as plt

cv = 100
scaler = MinMaxScaler()
le = LabelEncoder()
# dataset = load_iris()
dataset = pd.read_csv('./data/entropy6_v1.csv')
dataset = dataset.drop(labels='uuid', axis=1)
dataset['label'] = le.fit_transform(dataset['label'])
print(dataset['label'].value_counts())
print(le.classes_)
# print(le.feature_names_in_)
k = len(le.classes_)

y = dataset['label']
unique_num, cnt = (np.unique(y, return_counts=True))
print(min(cnt))


df = pd.DataFrame()

for i in range(k):
    # print(dataset.loc[dataset['label'] == i])
    if i == 1:
        continue
    # mask =
    temp = dataset.loc[dataset['label'] == i].sample(n=300)
    # print(len(temp))
    df = df.append([temp])
# print(len(df))
df = df.drop(labels='Unnamed: 0', axis=1)
# print(df)
x_data = df.iloc[:,:-1]
# print(x_data.dtypes)
# print(x_data)
x_data = np.array(x_data)
# print(x_data)
# x_data = pd.to_numeric(x_data)
x_data = scaler.fit_transform(x_data)
y_data = df['label']
# print(np.mean(x_data, axis=0))
# print(np.var(x_data, axis=0))
# print(y_data.value_counts())

# kf = StratifiedKFold(n_splits=cv)
# print(kf.get_n_splits(x_data))

# types = ['classification', 'moon', 'blobs', 'circles']
# n_clusters = 3
# n_features = 2
# samples = 500
# n_clusters_per_class = 1
# types_num = 1
# idx, x, y = generate(type_=types[types_num], n_cluster=n_clusters, n_features=n_features, n_samples=samples,
#                      n_clusters_per_class=n_clusters_per_class)
# plot_data(x, y)
# x_data = dataset['data']
# y_data = dataset['target']
# label_names_ = dataset['target_names']

# x_data = x
# y_data = y.ravel()
# label_names_ = np.unique(y)

interval = [20]

label_mask = []
model4 = KMeans(n_clusters=6)
model4.fit_predict(x_data)
print('KMeans\'s RI', rand_score(y_data, model4.labels_))
print('KMeans\'s ARI', adjusted_rand_score(y_data, model4.labels_))

for label in np.unique(y_data):
    mask = list(np.where(y_data == label))
    label_mask.append(mask)

for itv in interval:
    # epsilon = np.linspace(0.5, 2, itv)
    epsilon = np.linspace(0.0001, 1, itv)
    min_samples = np.arange(2, len(x_data), 10)
    # DIR_PATH = r'C:\Users\YoungHo\Documents\Cloud\ML_Project\HRclustering\data'
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
            start = time.time()
            model1.fit(x_data)
            end = time.time()
            duration = end - start
            print(duration)
            # print(model1.labels_)
            print(model1.labels_)
            print(np.unique(model1.labels_, return_counts=True))
            dbscan_ARI.append(rand_score(y_data, model1.labels_))
            print('DBSCAN\'s RI: ', rand_score(y_data, model1.labels_))
            print('DBSCAN\'s ARI: ', adjusted_rand_score(y_data, model1.labels_))

            model2 = HR_clustering(X=x_data, tau=eps)
            start = time.time()
            model2.fit_predict(min_samples=min_sam)
            end = time.time()
            duration = end - start
            print(duration)
            print(model2.labels_)
            print(np.unique(model2.labels_, return_counts=True))
            print('HRDBSCAN\'s RI: ', rand_score(y_data, model2.labels_))
            print('HRDBSCAN\'s ARI: ', adjusted_rand_score(y_data, model2.labels_))
            hrdbscan_ARI.append(rand_score(y_data, model2.labels_))

            model3 = hdbscan.HDBSCAN(min_samples=int(min_sam), cluster_selection_epsilon=float(eps))
            start = time.time()
            model3.fit_predict(x_data)
            hdbscan_ARI.append(rand_score(y_data, model3.labels_))
            end = time.time()
            duration = end - start
            print(duration)
            print(model3.labels_)
            print(np.unique(model3.labels_, return_counts=True))
            print('HDBSCAN\'s RI: ', rand_score(y_data, model3.labels_))
            print('HDBSCAN\'s ARI: ', adjusted_rand_score(y_data, model3.labels_))




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
        #
        # FILE_NAME = 'eps_{:.3f}_min_sample_{}_{}.png'.format(eps, np.min(min_samples), np.max(min_samples))
        # PATH_ARI = os.path.join(DIR_PATH_ARI, FILE_NAME)
        # plt.plot(np.arange(len(min_samples)), dbscan_ARI, c='red', label='DBSCAN')
        # plt.plot(np.arange(len(min_samples)), hrdbscan_ARI, c='blue', label='HR-DBSCAN')
        # plt.plot(np.arange(len(min_samples)), hdbscan_ARI, c='green', label='HDBSCAN')
        # plt.ylabel('Adjusted Rand Index', fontsize=15)
        # plt.ylim(0.0, 1.05)
        # plt.xlabel('min samples', fontsize=15)
        # plt.title('test on epsilon {}'.format(eps), fontsize=20)
        # plt.grid()
        # plt.legend()
        # plt.savefig(PATH_ARI)
        # plt.close()
