from HR_TREE.HRHC import *
from HR_TREE.make_data import *
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":

    index, X, y = generate(type_='blobs', n_cluster=2, n_features=2, n_samples=500, n_clusters_per_class=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(X)
    # print(X.shape)
    # plot_data(X, y)
    cl_model = HRHC(X=X, index=index, tau=0.06)
    cl_model.fit_predict(min_samples=3)
    # print(cl_model.hierarchy_prototypes_index[1])
