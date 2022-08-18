from sklearn.datasets import make_classification, make_blobs, make_circles, make_moons

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def generate(type_, n_cluster=2, n_samples=500, n_features=2, n_clusters_per_class=1, random_state=42):

    index = None
    X = None
    y = None

    if type_ is 'classification':
        # generate classification data
        # n_features is up to 10?
        # X's shape (n_samples, n_features)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)

        DIR_PATH = os.getcwd() + r'\data'
        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_classification(n_samples, n_features, n_classes=n_cluster, n_informative=n_features,
                                   random_state=random_state, n_clusters_per_class=n_clusters_per_class, n_redundant=0, n_repeated=0)
        index = np.arange(n_samples)
        y = y.reshape(-1, 1)
        feature_list = []
        for i in range(n_features):
            feature_list.append('X_{}'.format(i+1))
        df_index = pd.DataFrame(np.arange(n_samples), columns=['index'])
        df_X = pd.DataFrame(X, columns=feature_list)
        df_y = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df_X], axis=1)
        df = pd.concat([df, df_y], axis=1)

        df.to_csv(PATH)

    elif type_ is 'moon':

        # n_features = 2,
        # X's shape (n_samples, 2)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)


        DIR_PATH = os.getcwd() + r'\data'

        if n_features > 2:
            return None, None, None
        DIR_PATH = r'C:\Users\YoungHo\Documents\Cloud\ML_Project\HR_TREE\data'

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_moons(n_samples=n_samples, random_state=random_state)
        index = np.arange(n_samples)
        y = y.reshape(-1, 1)

        df_index = pd.DataFrame(index, columns=['index'])
        df1 = pd.DataFrame(X, columns=['X1', 'X2'])
        df2 = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df1], axis=1)
        df = pd.concat([df, df2], axis=1)

        df.to_csv(PATH)

    elif type_ is 'blobs':

        # n_features = n_features,
        # X's shape (n_samples, n_features)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)

        DIR_PATH = os.getcwd() + r'\data'

        if n_features > 2:
            return None, None, None
        DIR_PATH = r'C:\Users\YoungHo\Documents\Cloud\ML_Project\HR_TREE\data'

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_blobs(centers=n_cluster, n_samples=n_samples, n_features=n_features, random_state=random_state)
        index = np.arange(n_samples)
        y = y.reshape(-1, 1)

        df_index = pd.DataFrame(index, columns=['index'])
        df1 = pd.DataFrame(X, columns=['X1', 'X2'])
        df2 = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df1], axis=1)
        df = pd.concat([df, df2], axis=1)

        df.to_csv(PATH)

    elif type_ is 'circles':
        # n_features = 2,
        # X's shape (n_samples, 2)
        # y's shape (n_samples,), so do reshape y' shape to (n_samples, 1)


        DIR_PATH = os.getcwd() + r'\data'

        if n_features > 2:
            return None, None, None
        DIR_PATH = r'C:\Users\YoungHo\Documents\Cloud\ML_Project\HR_TREE\data'

        FILE_NAME = type_ + '_' + str(n_samples) + '_' + str(n_cluster) + '_' + str(n_features) + '.csv'
        PATH = os.path.join(DIR_PATH, FILE_NAME)

        X, y = make_circles(n_samples=n_samples, random_state=random_state)
        index = np.arange(n_samples)
        y = y.reshape(-1, 1)

        df_index = pd.DataFrame(index, columns=['index'])
        df1 = pd.DataFrame(X, columns=['X1', 'X2'])
        df2 = pd.DataFrame(y, columns=['target'])
        df = pd.concat([df_index, df1], axis=1)
        df = pd.concat([df, df2], axis=1)

        df.to_csv(PATH)

    else:

        print('please check input')

    return index, X, y


def plot_data(X, y):


    color = ['red', 'blue', 'green']

    for i in np.unique(y):
        mask = np.where(y==i)
        plt.scatter(X[mask, 0], X[mask, 1], label=str(i), cmap=i)
    plt.legend()
    plt.show()


    if X.shape[1] == 2:
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.legend()
        plt.show()


if __name__ == '__main__':


    index, X, y = generate(type_='moon', n_cluster=3, n_features=2, n_samples=500, n_clusters_per_class=1)
    # print(X.shape)
    # print(y)
    plot_data(X, y)

    types = ['classification', 'moon', 'blobs', 'circles']

    for t in types:
        for n_feat in range(2,10):
            for n_cls in range(2, 5):
                index, X, y = generate(type_=t, n_cluster=n_cls, n_features=n_feat, n_samples=500, n_clusters_per_class=1)

