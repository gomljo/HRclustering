import numpy as np

from HRclustering.HRHC import *
from HRclustering.make_data import *
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
    cl_model = HRHC(X=X, tau=0.1, index=index)
    cl_model.fit_predict(min_samples=3)
    # print(X)
    # print(X.shape)
    # plot_data(X, y)

    # tau_ = np.linspace(1e-3, 1, 1000)
    # tau_ = 0.102
    # stage_ = 1
    # init_min_sam = 3
    # max_proto = 500
    #
    # n_proto = []
    # models = []
    # cnt_1 = 0
    # plt.figure(figsize=(20, 20))
    # axes = plt.axes(projection='3d')
    #
    # for sam in trange(init_min_sam, max_proto, 10):
    #     cl_model = HRHC(X=X, tau=tau_, index=index)
    #     cl_model.make_hierarchy_hr_test(cl_model.tau)
    #     cl_model.make_hierarchy_hr(tau=cl_model.tau, min_samples=sam)
    #     # print(len(self.prototypes))
    #     n_proto.append(len(cl_model.prototypes))
    #     expected_proto = len(cl_model.prototypes) // 2
    #
    #     models = [cl_model] * (max_proto - sam)
    #
    #     n_proto_3 = [] * (max_proto - init_min_sam)
    #     end_idx = 0
    #     for sam3 in trange(sam, max_proto, 10):
    #         # print(sam3)
    #         nproto = models[(sam3 - sam)].make_hierarchy_hr(tau=models[(sam3 - sam)].tau, min_samples=sam3)
    #         # if nproto == 0:
    #         #     end_idx = sam3
    #         #     break
    #         n_proto_3[cnt_1].append(nproto)
    #         cnt_1 += 1
    #         sleep(0.01)
    # axes.plot3D(, np.arange(len(n_proto_3)), n_proto_3)
    # plt.xticks(np.arange(0, len(n_proto_3)), np.arange(sam, end_idx + 1, 10), rotation=45)
    # plt.yticks(np.arange(0, max(n_proto_3) + 1, 10), np.arange(0, max(n_proto_3) + 1, 10))
    # plt.grid()
    # plt.show()

    cl_model.fit_predict(min_samples=3)
    # cl_model.report_change_tau()

    '''
    Under the code how to experiment proper tau and min_samples
    key is deepcopy because HRHC class instance share variables that changes each iterations
    So,we use deep copy method for preventing variables value changes 
    '''
    # max_proto = 500
    # best_model = []
    #
    # tau = np.linspace(1e-3, 1.0, 1000)
    # best_model_stage1 = None
    # for idx1, t_ in enumerate(tau):
    #     models = deepcopy(self)
    #
    #     models.make_hierarchy_hr(tau=t_, min_samples=min_samples)
    #
    #     # print(len(models[idx1].prototypes))
    #     if len(models.hierarchy_prototypes[0]) == max_proto:
    #         best_model_stage1 = deepcopy(models)
    #         tau = tau[idx1:]
    #         print('second stage is started with tau {}'.format(t_))
    #         break
    # expected_proto = int(len(best_model_stage1.hierarchy_prototypes[0]) // 2)
    # n_proto = len(best_model_stage1.hierarchy_prototypes[0])
    # cnt = 0
    # best_model_each = deepcopy(best_model_stage1)
    # best_model.append(deepcopy(best_model_stage1))
    #
    # while n_proto >= 1:
    #     # if cnt > 500:
    #     #     break
    #     self.tau = tau[cnt]
    #     min_sample = np.arange(min_samples, max_proto)
    #
    #     model = None
    #     print('iter {} started'.format(cnt + 1))
    #     for idx2, min_sam in enumerate(tqdm(min_sample)):
    #         model = deepcopy(best_model_each)
    #         model.make_hierarchy_hr(tau=model.tau, min_samples=min_sam)
    #
    #         if (len(model.prototypes) <= expected_proto) and (len(model.prototypes) != 0):
    #             best_model_each = deepcopy(model)
    #             n_proto = len(model.prototypes)
    #             break
    #         else:
    #             model = None
    #         sleep(0.001)
    #
    #     cnt += 1
    #     if model is None:
    #         print('iter {} needs to increase tau value'.format(cnt + 1))
    #         continue
    #     else:
    #         best_model.append(deepcopy(best_model_each))
    #     print(len(best_model_each.prototypes))
    #     print('expected number of prototype >= ', len(best_model_each.prototypes) // 2)
    #     expected_proto = int(len(best_model_each.prototypes) // 2)
    #     if expected_proto <= 0:
    #         expected_proto = 1