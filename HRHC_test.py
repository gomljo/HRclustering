import numpy as np

from HRHC import *
from make_data import *
from sklearn.preprocessing import MinMaxScaler


def test_B_1(tau, stage):

    DIR_PATH = os.getcwd() + r'\varying tau'
    for i in range(1, stage):
        n_proto = []
        for t_ in tau:
            cl_model = HRHC(X=X, index=index, min_samples=3)
            for _ in range(0, i):
                cl_model.make_hierarchy_hr_test(tau=t_)
            print(len(cl_model.hyper_rectangles[0]))
            n_proto.append(len(cl_model.hyper_rectangles[0]))
            sleep(0.01)

        FILE_NAME = 'Hierarchy {} tau {} ~ {} min_samples {}'.format((i + 1), np.min(tau), np.max(tau), 3)
        PATH = os.path.join(DIR_PATH, FILE_NAME)
        plt.plot(tau, n_proto)
        plt.xlabel('tau', fontsize=15)
        plt.ylabel('Num. of prototypes', fontsize=15)
        plt.title('Num. of prototypes varying tau {} ~ {}'.format(np.min(tau), np.max(tau)), fontsize=20)
        plt.savefig(PATH)


if __name__ == "__main__":

    index, X, y = generate(type_='blobs', n_cluster=2, n_features=2, n_samples=500, n_clusters_per_class=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(X)
    # print(X.shape)
    # plot_data(X, y)
    tau_ = np.linspace(1e-3, 1, 1000)
    stage_ = 5
    test_B_1(tau_, stage_)
    # cl_model.fit_predict(min_samples=3)
    # cl_model.report_change_tau()