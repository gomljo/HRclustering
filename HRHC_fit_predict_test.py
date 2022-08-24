import numpy as np

from HRclustering.HRHC import *
from HRclustering.make_data import *
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange, tqdm


if __name__ == "__main__":

    idx, x, y = generate(type_='blobs', n_cluster=2, n_features=2, n_samples=500, n_clusters_per_class=1)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    # print(X)
    # print(X.shape)
    # plot_data(X, y)

    # tau_ = np.linspace(1e-3, 1, 1000)
    tau_ = [np.linspace(0.10101010101010102, 1, 1000)]
    stage_ = 1
    min_samples_ = 3
    models_step_1 = []
    models_step_2 = []
    models_step_3 = []
    models_step_4 = []
    models_step_5 = []

    for t_ in tau_:
        model = HRHC(X=x, tau=t_, index=idx, min_samples=min_samples_)
        model.fit_predict_test(tau=t_, min_samples=min_samples_, stage=stage_)
        models_step_1.append(model)



    for mod in models_step_1:
        max_proto = mod.prototypes
        for proto in range(min_samples_, max_proto):
            mod.fit_predict_test(mod.tau, min_samples=proto,stage=stage_)
            models_step_2.append(mod)


    # cl_model.fit_predict(min_samples=3)
    # cl_model.report_change_tau()