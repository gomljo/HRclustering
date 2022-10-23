from HRclustering.HRHC import *
from HRclustering.make_data import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score


if __name__ == "__main__":

    index, X, y = generate(type_='blobs', n_cluster=3, n_features=2, n_samples=500, n_clusters_per_class=1)

    print(np.unique(y, return_counts=True))
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = y.ravel()
    plot_data(X, y)
    cl_model = HRHC(X=X, index=index, tau=0.05, min_samples=4)
    cl_model.fit_predict(min_samples=3)
    # print(cl_model.labels_)
    y_pred = cl_model.labels_.flatten()
    print(adjusted_rand_score(y, y_pred))
    # np.unique()