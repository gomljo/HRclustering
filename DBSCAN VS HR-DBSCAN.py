# from sklearn.cluster import DBSCAN
# from LearnKit.AnomalyDetection.HyperRectangleDBSCAN import HyperRectangle, HyperRectangleClustering
# from sklearn.datasets import load_iris, load_wine
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import classification_report, roc_curve, matthews_corrcoef, roc_auc_score
#
# import numpy as np
#
#
# def data_loader(d_name='iris'):
#     le = LabelEncoder()
#     data = None
#     if d_name is 'iris':
#         data = load_iris()
#
#     elif d_name is 'wine':
#         data = load_wine()
#     if data is None:
#         print('Please check data name(d_name)')
#     # x_train, x_test, y_train, y_test = train_test_split(data)
#
#     return data['data'], data['target']
#
#
# def dbscan_learning(epsilon, min_samples, metric='euclidean', d_name='iris', cv=5):
#     x_data, y_data = data_loader(d_name)
#
#     kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
#     for train_indexes, test_indexes in kf.split(x_data, y_data):
#         X_train = x_data[train_indexes]
#         y_train = y_data[train_indexes]
#         X_test = x_data[test_indexes]
#         y_test = y_data[test_indexes]
#         model = DBSCAN(eps=epsilon, min_samples=min_samples, metric=metric)
#         model.fit(X_train)
#         train_label = model.labels_
#         print(train_label)
#         y_pred = model.fit_predict(X_test)
#         print(y_pred)
#
#
# def hr_dbscan_learning(cv=5, d_name='iris'):
#
#     x_data, y_data = data_loader(d_name)
#
#     kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
#     for train_indexes, test_indexes in kf.split(x_data, y_data):
#         X_train = x_data[train_indexes]
#         y_train = y_data[train_indexes]
#         X_test = x_data[test_indexes]
#         y_test = y_data[test_indexes]
#
#         classifier = HyperRectangleClustering(X_train, tau=0.3)
#         y_train_pred = classifier.fit_predict(min_samples=2)
#         print(y_train_pred)
#         y_test_pred = classifier.predict(X_test)
#         print(y_test_pred)
#
#
# if __name__ == '__main__':
#
#     print('dbscan result')
#     dbscan_learning(epsilon=0.7, min_samples=3)
#     print('hr dbscan result')
#     hr_dbscan_learning()




from itertools import combinations
from HR_TREE.dbscan_analysis import *

import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def two_dim_plot(data, label, feature_names, label_names, model=None, epsilon_=None, min_samples_=None, tau_=None, data_name=None):

    data_labeled_idx = dict()
    unique_label = list(set(label))

    if len(unique_label) <= 0:
        print('please check label')
        return 0
    cmap = get_cmap(len(unique_label), name='Accent')

    for i in unique_label:
        data_labeled_idx[i] = np.where(label == i)
        print(cmap(i))
    feature_list = feature_names
    feature_list_idx = dict()
    for idx, feature in enumerate(feature_list):
        feature_list_idx[feature] = idx

    two_dim_feature = list(combinations(feature_list, 2))
    two_dim_feature_idx = list(combinations(feature_list_idx.values(), 2))

    for feature_name, feature_idx in zip(two_dim_feature, two_dim_feature_idx):
        # print(feature_name, feature_idx)
        for idx, data_idx in enumerate(data_labeled_idx.values()):
            plt.scatter(data[data_idx][:, feature_idx[0]], data[data_idx][:, feature_idx[1]], color=cmap(idx), label=label_names[idx])
            if idx == 0:
                plt.xlabel(feature_name[0], fontsize=15)
                plt.ylabel(feature_name[1], fontsize=15)
            plt.legend(fontsize=12)

        if model is not None:
            if model is 'DBSCAN':
                dir_path = r'D:\DBSCAN VS HRDBSCAN\DBSCAN'
                file_name = r'\DBSCAN ' + 'epsilon ' + str(epsilon_) + 'min_samples ' + str(min_samples_)
                path = os.path.join(dir_path, file_name)
                if not os.path.exists(path):
                    os.mkdir(path)
                plt.title(model + '\'s parameter\n ' + 'epsilon: ' + str(epsilon_) + ', ' + 'min_samples: ' + str(min_samples_))
            else:
                path = r'D:\DBSCAN VS HRDBSCAN\HRDBSCAN' + r'\HRDBSCAN' + 'tau ' + str(tau_) + 'min_samples ' + str(min_samples_)
                if not os.path.exists(path):
                    os.mkdir(path)
                plt.title(model + '\'s parameter\n ' + 'tau: ' + str(tau_) + ', ' + 'min_samples: ' + str(min_samples_))
            plt.savefig(path + '/' + model + '_' + feature_name[0] + ' vs ' + feature_name[1] + '.png')
        else:
            path = r'D:\DBSCAN VS HRDBSCAN' + r'\Dataset'+'(' + data_name+ ')'
            if not os.path.exists(path):
                os.mkdir(path)
            plt.savefig(path + '/' + feature_name[0] + ' vs ' + feature_name[1] + '.png')
        plt.close()


def compare_plot(A, B, data_name):

    analysis_label = ['Num. of clusters', 'Num. of noise', 'Num. of core samples', 'Homogeneity score',
                      'Completeness score', 'V measure', 'adjusted rand index', 'adjusted mutual information',
                      'Silhouette coefficient']

    analysis_val_A = [A.n_clusters_, A.n_noise_, A.n_core_samples, A.homogeneity_score,
                    A.completeness_score, A.v_measure,
                    A.adjusted_rand_index, A.adjusted_mutual_information, A.silhouette_coef]
    analysis_val_B = [B.n_clusters_, B.n_noise_, B.n_core_samples, B.homogeneity_score,
                      B.completeness_score, B.v_measure,
                      B.adjusted_rand_index, B.adjusted_mutual_information, B.silhouette_coef]

    dir_path = r'D:\DBSCAN VS HRDBSCAN\Compare epsilon' + data_name + str(min(A.radius)) + '~' + str(
            max(A.radius)) + 'min_samples ' + str(min(A.min_samples)) + '~' + str(max(A.min_samples))

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    print('plotting.....')

    for idx, min_samples_ in enumerate(tqdm(A.min_samples)):
        time.sleep(0.01)
        for idx2, label in enumerate(analysis_label):

            plt.plot(np.arange(A.num_r), analysis_val_A[idx2][:, idx], c='red', label='DBSCAN')
            plt.plot(np.arange(A.num_r), analysis_val_B[idx2][:, idx], c='blue', label='HRDBSCAN')
            plt.xticks([0, 200, 400, 600, 800], ['0.1', '0.3', '0.5', '0.7', '0.9'])

            max_val_B = np.max(analysis_val_B[idx2])
            max_val_A = np.max(analysis_val_A[idx2])
            if max_val_B and max_val_A < 1:

                if max_val_B > max_val_A:
                    max_val = max_val_B

                else:
                    max_val = max_val_A
                max_val += max_val * 0.3
            else:
                max_val_B.astype(np.int64)
                max_val_A.astype(np.int64)
                max_val = max(max_val_B, max_val_A)
                max_val += 3
            # y_max = max(np.max(analysis_val_B[idx2], np.max(analysis_val_A[idx2])))

            plt.ylim(0, max_val)

            plt.xlabel('radius', fontsize=20)
            plt.title(
                    'model\'s parameter\n ' + 'min_samples: ' + str(min_samples_))
            plt.grid()
            plt.ylabel(label)
            plt.legend()
            file_name = label + r' min_samples ' + str(min_samples_) + '.png'
            path = os.path.join(dir_path, file_name)
            plt.savefig(path)
            plt.close()


if __name__ == '__main__':
    from LearnKit.AnomalyDetection.HyperRectangleDBSCAN import HyperRectangleClustering
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    data_name = ['wine', 'breast_cancer']
    load_data = [load_wine(), load_breast_cancer()]

    for idx, load in enumerate(load_data):

        dataset = load
        x_data = dataset['data']
        y_data = dataset['target']
        label_names_ = dataset['target_names']
        # two_dim_plot(x_data, y_data, dataset['feature_names'], label_names_, data_name='iris')

        x_data = scaler.fit_transform(x_data)

        epsilon = np.arange(0.1, 1, 0.001)
        # print(epsilon)
        min_samples = np.arange(2, 11, 1)

        model_names = ['dbscan', 'hrdbscan']

        tool_A = Analysis(X=x_data, model_name=model_names[1], labels_true=y_data)
        tool_A.get_metrics(radius=epsilon, min_samples=min_samples)

        tool_B = Analysis(X=x_data, model_name=model_names[0], labels_true=y_data)
        tool_B.get_metrics(radius=epsilon, min_samples=min_samples)

        compare_plot(tool_A, tool_B, data_name[idx])