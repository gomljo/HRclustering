import numpy as np

from HRHC import *
from make_data import *
from sklearn.preprocessing import MinMaxScaler
from tqdm import trange, tqdm


def test_B_1(tau, stage):

    # proof for consideration 1.A.i
    DIR_PATH = os.getcwd() + r'\varying tau'
    for i in range(1, stage):
        n_proto = []
        for t_ in tau:
            cl_model = HRHC(X=X, index=index, min_samples=3)
            for _ in range(0, i):
                cl_model.make_hierarchy_hr_test(tau=t_)
            n_proto.append(cl_model.prototypes)

            sleep(0.01)

        FILE_NAME = 'Hierarchy {} tau {} ~ {} min_samples {}.png'.format((i), np.min(tau), np.max(tau), 3)
        PATH = os.path.join(DIR_PATH, FILE_NAME)
        plt.plot(tau, n_proto)
        plt.xlabel('tau', fontsize=15)
        plt.ylabel('Num. of prototypes', fontsize=15)
        plt.title('Num. of prototypes varying tau {} ~ {}'.format(np.min(tau), np.max(tau)), fontsize=20)
        plt.savefig(PATH)
        plt.grid()
        plt.close()


def test_B_2(tau, stage):

    DIR_PATH = os.getcwd() + r'\varying tau'
    # 모델에 대한 데이터가 필요하므로 이전 단계의 모델 객체를 리스트 형태로 저장 후
    # 다음 단계에서 사용해야 할 듯?

    for t_ in (tau):
        n_proto = []
        cl_model = HRHC(X=X, index=index, min_samples=3, tau=t_)
        init_min_sample = 3
        max_proto = len(cl_model.prototypes)
        for step in range(1, stage):

            for n_sam in range(init_min_sample, max_proto):
                # print(n_sam)
                cl_model = HRHC(X=X, index=index, min_samples=3)
                cl_model.make_hierarchy_hr_test(tau=t_, min_samples=n_sam)
                # print(len(cl_model.prototypes))
                if len(cl_model.prototypes) > 0:
                    n_proto.append(len(cl_model.prototypes))
                else:
                    n_proto.append(0)
                # sleep(0.01)
        # print(n_proto)

            FILE_NAME = 'Hierarchy_{}_tau_{.4f}_min_samples_{}_{}.png'.format(step, t_, 3, np.max(len(cl_model.prototypes)))
            PATH = os.path.join(DIR_PATH, FILE_NAME)
            plt.plot(np.arange(max_proto - init_min_sample), n_proto)
            plt.xlabel('tau', fontsize=15)
            plt.ylabel('Num. of prototypes', fontsize=15)
            plt.title('Num. of prototypes varying tau {.4f} min_samples_{} ~ {}'.format(t_, 3, np.max(len(cl_model.prototypes)), fontsize=20))
            plt.savefig(PATH)
            plt.close()


if __name__ == "__main__":

    index, X, y = generate(type_='blobs', n_cluster=2, n_features=2, n_samples=500, n_clusters_per_class=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(X)
    # print(X.shape)
    # plot_data(X, y)

    # tau_ = np.linspace(1e-3, 1, 1000)
    tau_ = [0.10101010101010102]
    stage_ = 1

    test_B_2(tau_, stage_)
    # cl_model.fit_predict(min_samples=3)
    # cl_model.report_change_tau()