from HRclustering.HRHC import *
from HRclustering.make_data import *
from sklearn.preprocessing import MinMaxScaler


if __name__ == "__main__":

    index, X, y = generate(type_='classification', n_cluster=2, n_features=2, n_samples=500, n_clusters_per_class=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    cl_model = HRHC(X=X, index=index, tau=0.105)
    cl_model.fit_predict(min_samples=3)

    np.unique()