from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import make_moons, make_blobs, make_classification, make_circles

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
le = LabelEncoder()


def Data_loader_Toy(type_=None, is_scale=False):
    X = None
    y = None
    name = type_
    if type_ is 'iris':
        dataset = load_iris()
        X = dataset['data']
        y = dataset['target']

    elif type_ is 'wine':
        dataset = load_wine()
        X = dataset['data']
        y = dataset['target']

    elif type_ is 'breast_cancer':
        dataset = load_breast_cancer()
        X = dataset['data']
        y = dataset['target']

    elif type_ is 'glass':
        dataset = pd.read_csv('./data/glass.csv')
        X = dataset.iloc[:,:-1]
        y = dataset['Type']

    elif type_ is 'bean':
        dataset = pd.read_csv('./data/Dry_Bean.csv')
        X = dataset.iloc[:, :-1]
        y = dataset['Class']
        y = le.fit_transform(y)

    elif type_ is 'raisin':
        dataset = pd.read_csv('./data/raisin.csv')
        X = dataset.iloc[:, :-1]
        y = dataset['Class']
        y = le.fit_transform(y)

    elif type_ is 'yeast':
        dataset = pd.read_csv('./data/yeast.csv')
        X = dataset.iloc[:, :-1]
        y = dataset['name']
        y = le.fit_transform(y)

    if is_scale:
        X = scaler.fit_transform(X)

    return name, X, y


def Data_loader_IoT(path, is_scale, class_per_sample=300):
    X = None
    y = None

    if '.csv' in path:
        dataset = pd.read_csv(path)
        # print(dataset.shape[0])
        dataset = dataset.drop(labels='uuid', axis=1)
        dataset = dataset.drop(labels='Unnamed: 0', axis=1)
        # print(dataset)

        X = dataset.iloc[:, :-1]

        malware_map = {
            'Benign': 0,
            'Agent' : 1,
            'Dofloo' : 2,
            'Gafgyt' : 3,
            'Mirai' : 4,
            'Nyadrop' : 5,
            'Tsunami' : 6
        }

        dataset['label'] = dataset['label'].map(malware_map)

        unique_num, cnt = (np.unique(dataset['label'], return_counts=True))
        print(np.unique(dataset['label'], return_counts=True))
        # print(min(cnt))

        if class_per_sample:
            # print('1')
            # if class_per_sample > min(cnt):
            #     print('please check minimum samples per classes')
            #     return None, None

            k = len(unique_num)
            df = pd.DataFrame()

            for i in range(k):
                # print(dataset.loc[dataset['label'] == i])
                # if i == 1:
                #     continue
                if i in [0, 2, 4]:
                    print(i)
                    temp = dataset.loc[dataset['label'] == i]
                    if temp.shape[0] < class_per_sample:
                        # print('not enough')
                        # print(cnt[i])
                        temp = temp.sample(n=cnt[i], random_state=42)
                    else:
                        temp = temp.sample(n=class_per_sample, random_state=42)
                    # print(temp.shape[0])
                    df = df.append([temp])
            x_data = df.iloc[:, :-1]
            # print(df)
            X = np.array(x_data)
            y = df['label']

    if is_scale:
        X = scaler.fit_transform(X)

    return X, y


def Data_loader_synthetic(type_, n_cluster=2, n_samples=500, n_features=2, n_clusters_per_class=1, random_state=42, is_scale=False):
    index = np.arange(n_samples)
    X = None
    y = None
    df = None
    DIR_PATH = os.getcwd() + r'\data'
    PATH = None
    if type_ is 'classification':
        # generate classification data
        # n_features is up to 10?
        # X's shape (n_samples, n_features)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_classification(n_samples, n_features, n_classes=n_cluster, n_informative=n_features,
                                   random_state=random_state, n_clusters_per_class=n_clusters_per_class, n_redundant=0,
                                   n_repeated=0)

        y = y.reshape(-1, 1)

        feature_list = []
        for i in range(n_features):
            feature_list.append('X_{}'.format(i + 1))
        df_index = pd.DataFrame(np.arange(n_samples), columns=['index'])
        df_X = pd.DataFrame(X, columns=feature_list)
        df_y = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df_X], axis=1)
        df = pd.concat([df, df_y], axis=1)

    elif type_ is 'moon':

        # n_features = 2,
        # X's shape (n_samples, 2)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)

        if n_features > 2:
            print('Please check feature dimension. n_features <= 2 required')
            return None, None

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_moons(n_samples=n_samples, random_state=random_state)

        y = y.reshape(-1, 1)

        df_index = pd.DataFrame(index, columns=['index'])
        df1 = pd.DataFrame(X, columns=['X1', 'X2'])
        df2 = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df1], axis=1)
        df = pd.concat([df, df2], axis=1)

    elif type_ is 'blobs':

        # n_features = n_features,
        # X's shape (n_samples, n_features)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)
        if n_features > 2:
            return None, None

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_blobs(centers=n_cluster, n_samples=n_samples, n_features=n_features, random_state=random_state)

        y = y.reshape(-1, 1)

        df_index = pd.DataFrame(index, columns=['index'])
        df1 = pd.DataFrame(X, columns=['X1', 'X2'])
        df2 = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df1], axis=1)
        df = pd.concat([df, df2], axis=1)

    elif type_ is 'circles':
        # n_features = 2,
        # X's shape (n_samples, 2)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)

        if n_features > 2:
            return None, None

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_circles(n_samples=n_samples, random_state=random_state)

        y = y.reshape(-1, 1)

        df_index = pd.DataFrame(index, columns=['index'])
        df1 = pd.DataFrame(X, columns=['X1', 'X2'])
        df2 = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df1], axis=1)
        df = pd.concat([df, df2], axis=1)

    else:
        print('please check input')

    if is_scale:
        df.iloc[:,:-1] = scaler.fit_transform(df.iloc[:, :-1])
        X = scaler.fit_transform(X)

    df.to_csv(PATH)

    return X, y


def plot_data(x_, label):

    if x_.shape[1] == 2:
        plt.scatter(x_[:, 0], x_[:, 1], c=label)
        plt.legend()
        plt.show()
