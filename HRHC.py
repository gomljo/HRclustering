
# import self defined class
from HRclustering.Hyper_Rectangle import HyperRectangle

# import utility libraries
from collections import deque
from tqdm import tqdm, trange
from copy import deepcopy
from time import sleep

import os
import numpy as np
import matplotlib.pyplot as plt


class HRHC:

    def __init__(self, X=None, tau=0.1, index=None, min_samples=3):
        self.X = X

        self.tau = tau
        self.min_samples = min_samples

        self.labels_ = np.zeros(self.X.shape[0])
        self.core_sample_indices_ = None

        self.hyper_rectangles = deque()
        self.in_data = set([])

        self.hierarchy_prototypes = deque()
        self.hierarchy_prototypes.appendleft(X)
        self.hierarchy_prototypes_index = deque()
        self.hierarchy_prototypes_index.appendleft(index)
        self.do_change = False

    def find_noise(self, x, index, min_samples, is_merge=False):

        hr = HyperRectangle(x=x, tau=self.tau, x_index=index, is_merge=is_merge)
        dup_index = list()
        # print(index)
        for idx, neighbor in enumerate(self.hierarchy_prototypes[0]):

            if index == idx:
                continue
            elif idx in self.in_data:
                continue
            dup_ind, is_include = hr.is_include(neighbor, idx)
            if is_include:
                # print(idx)
                dup_index.append(dup_ind)
                self.in_data.add(dup_ind)

        # print(dup_index)
        if len(hr.covered_data_indexes) < min_samples:
            # print('1')
            hr.y = -1
            return None, []

        return hr, dup_index

    # @staticmethod
    def hr_update(self, hr):

        # for in_hr_idx in hr.covered_data_indexes:
        # print(list(hr.covered_data_indexes))
        hr.hr_mid = np.mean(self.X[list(hr.covered_data_indexes)], axis=0)
        hr.hr_max = np.max(self.X[list(hr.covered_data_indexes)], axis=0)
        hr.hr_min = np.min(self.X[list(hr.covered_data_indexes)], axis=0)
        hr.R = (hr.hr_max - hr.hr_min) / 2.0

    def get_prototypes(self):
        prototypes = list()
        prototypes_index = list()

        for hr in self.hyper_rectangles[0]:
            prototypes.append(hr.hr_mid)
            prototypes_index.append(hr.x_idx)

        prototypes = np.array(prototypes)
        prototypes_index = np.array(prototypes_index)

        self.hierarchy_prototypes.appendleft(prototypes)
        self.hierarchy_prototypes_index.appendleft(prototypes_index)

    def make_hierarchy_hr(self, tau, min_samples=3, is_merge=False):

        hyper_rectangles = list()
        if self.tau != tau:
            self.tau = tau
        if self.min_samples != min_samples:
            self.min_samples = min_samples

        for index, x in enumerate(self.hierarchy_prototypes[0]):
            if index in self.in_data:
                continue

            hr, duplicated_idx = self.find_noise(x=x, index=index, min_samples=min_samples, is_merge=is_merge)

            if hr is not None:
                # print('before: ', hr.hr_mid)
                print(duplicated_idx)
                if is_merge:
                    print(index)
                    hr.covered_data_indexes.update(self.hyper_rectangles[0][index].covered_data_indexes)
                    for dup in duplicated_idx:
                        print(self.hyper_rectangles[0][dup].covered_data_indexes)
                        hr.covered_data_indexes.update(self.hyper_rectangles[0][dup].covered_data_indexes)

                self.hr_update(hr)
                # print('after: ',hr.hr_mid)

                hyper_rectangles.append(hr)

        # print(len(hyper_rectangles))
        self.hyper_rectangles.appendleft(hyper_rectangles)
        self.get_prototypes()
        self.in_data = set([])

    def pdf_multivariate_gauss(x_range, mu, cov):
        """
        Calculate the multivariate normal density (pdf)

        Keyword arguments:
            x = numpy array of a "d x 1" sample vector
            mu = numpy array of a "d x 1" mean vector
            cov = "numpy array of a d x d" covariance matrix
        '"""
        assert (mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
        assert (x_range.shape[0] > x_range.shape[1]), 'x must be a row vector'
        assert (cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
        assert (mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
        assert (mu.shape[0] == x_range.shape[0]), 'mu and x must have the same dimensions'

        part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
        part2 = (-1 / 2) * ((x_range - mu).T.dot(np.linalg.inv(cov))).dot((x_range - mu))
        return float(part1 * np.exp(part2))

    def fit_predict(self, min_samples=3):

        self.make_hierarchy_hr(tau=self.tau, min_samples=3)
        # print(len(self.hyper_rectangles[0]))
        n_d = 0
        for idx, hr in enumerate(self.hyper_rectangles[0]):
            print(idx, len(hr.covered_data_indexes))
            n_d += len(hr.covered_data_indexes)
            # print(self.X[list(hr.covered_data_indexes)])
            plt.scatter(self.X[list(hr.covered_data_indexes)][:,0], self.X[list(hr.covered_data_indexes)][:, 1], cmap=idx, label=idx)
        plt.legend()
        plt.show()
        print(n_d)



        # for hr in self.hyper_rectangles[0]:
            # mu = np.mean(self.X[list(hr.covered_data_indexes)], axis=0).reshape(-1, 1)
            # cov = np.cov(self.X[list(hr.covered_data_indexes)])
            # print(hr.covered_data_indexes)
            # for hr_neigh in self.hyper_rectangles[0]:
            #
            #     mu_neigh = np.mean(self.X[list(hr.covered_data_indexes)], axis=0).reshape(-1, 1)
            #     cov_neigh = np.cov(self.X[list(hr.covered_data_indexes)])

        # print(n_d)
        # print(self.hyper_rectangles[0])
        label = 0
        self.make_hierarchy_hr(tau=self.tau+0.02, min_samples=2, is_merge=True)
        # print(len(self.hyper_rectangles[0]))
        n_d = 0
        for idx, hr in enumerate(self.hyper_rectangles[0]):
            print(idx, len(hr.covered_data_indexes))
            n_d += len(hr.covered_data_indexes)
            # print(self.X[list(hr.covered_data_indexes)])
            plt.scatter(self.X[list(hr.covered_data_indexes)][:, 0], self.X[list(hr.covered_data_indexes)][:, 1],
                        cmap=idx, label=idx)
            plt.show()
        plt.legend()
        plt.show()
        print(n_d)
        # print(len(self.hyper_rectangles[0]))
        # print(self.hyper_rectangles[0])
        # label = 0
        # for hr in self.hyper_rectangles[0]:
        #
        #     if hr.y != -1:
        #         hr.y = label
        #     self.labels_[list(hr.covered_data_indexes)] = hr.y
        #     label += 1


    # def fit_predict(self, min_samples=3):
    #
    #     interval = 100
    #
    #     min_sam = min_samples
    #     # max_sam = len(self.hierarchy_prototypes)
    #     stage = 1
    #
    #     while len(self.hierarchy_prototypes[0]) != 1:
    #         tau_ = np.linspace(self.tau, 1, interval)
    #         print('stage {} is progressed...'.format(stage))
    #
    #         self.make_hierarchy_hr(tau=self.tau, min_samples=min_sam)
    #
    #         if (len(self.hierarchy_prototypes[0]) == len(self.hierarchy_prototypes[1])) or (len(self.hierarchy_prototypes[0]) == 0):
    #             print('calibrate tau...')
    #             # print(self.hyper_rectangles)
    #             self.hierarchy_prototypes.popleft()
    #             self.hierarchy_prototypes_index.popleft()
    #             self.hyper_rectangles.popleft()
    #
    #             is_change = False
    #             for idx, t_ in enumerate(tau_):
    #                 # print(t_)
    #                 # print(self.hyper_rectangles[0])
    #
    #                 self.make_hierarchy_hr(tau=t_, min_samples=min_sam)
    #
    #                 if len(self.hierarchy_prototypes[0]) != len(self.hierarchy_prototypes[1]) and (len(self.hierarchy_prototypes[0]) !=0):
    #                     print('Find proper tau value')
    #                     print(self.hyper_rectangles[0])
    #                     print('tau: {}'.format(t_))
    #                     print()
    #                     self.tau = t_
    #                     is_change = True
    #                     break
    #                 else:
    #                     self.hierarchy_prototypes.popleft()
    #                     self.hierarchy_prototypes_index.popleft()
    #                     self.hyper_rectangles.popleft()
    #             if not is_change:
    #                 print('Number of Prototype does not changed in stage {}.'.format(stage))
    #                 print('Error: Model needs to changing self.tau = {} or self.min_samples = {}.'.format(self.tau, self.min_samples))
    #                 break
    #
    #         print('-' * 50)
    #         print('Num. of prototypes: {}'.format(len(self.hierarchy_prototypes[0])))
    #         cnt = 0
    #         for idx, hr in enumerate(self.hyper_rectangles[0]):
    #             print('Num. of data in hyper rectangle {}: {}'.format(idx, len(hr.covered_data_indexes)))
    #             print('prototype in hyper rectangle {}: index {}, data {}'.format(idx, hr.x_idx, hr.hr_mid))
    #             cnt += len(hr.covered_data_indexes)
    #         print('total num. of data in hyper rectangles: {}'.format(cnt))
    #         print('Next step\'s prototype indices')
    #         print(self.hierarchy_prototypes_index[0])
    #         print('Next step\'s prototypes')
    #         print(self.hierarchy_prototypes[0])
    #         print('-' * 50)
    #         # self.tau *= 2
    #         stage += 1

    def predict(self, X):

        n_samples = X.shape[0]
        y_new = np.ones(shape=n_samples, dtype=int) * -1

        for i, data in enumerate(X):
            new_hr = HyperRectangle(x=data, tau=self.tau, x_index=i)

            for j, hr in enumerate(self.hyper_rectangles):
                if hr.y == -1:
                    continue

                if new_hr.is_include(hr.hr_mid, None):
                    y_new[i] = hr.y

                    break

        return y_new


class HRHC_TEST:

    def __init__(self, X=None, tau=0.1, index=None, min_samples=3):

        self.prototypes = X
        self.X = X

        self.tau = tau
        self.min_samples = min_samples

        self.labels_ = None
        self.core_sample_indices_ = None

        self.hyper_rectangles = deque()
        self.in_data = set([])

        self.index = index
        self.hierarchy_prototypes = deque()
        self.hierarchy_prototypes_index = deque()
        self.do_change = False

    def find_noise_test(self, x, index, tau, min_samples):

        hr = HyperRectangle(x=x, tau=tau, x_index=index)

        for idx, neighbor in zip(self.index, self.prototypes):

            if index == idx:
                continue
            # elif idx in self.in_data:
            #     continue
            dup_ind = hr.is_include(neighbor, idx)
            # if dup_ind is not None:
            #     self.in_data.add(dup_ind)

        if len(hr.covered_data_indexes) < min_samples:
            hr.y = -1
            return None

        return hr

    def get_prototypes_test(self, change=True):
        prototypes = list()
        prototypes_index = list()

        for hr in self.hyper_rectangles[0]:
            prototypes.append(hr.hr_mid)
            prototypes_index.append(hr.x_idx)

        prototypes = np.array(prototypes)
        prototypes_index = np.array(prototypes_index)

        self.prototypes = prototypes

        self.hierarchy_prototypes.appendleft(prototypes)
        self.hierarchy_prototypes_index.appendleft(prototypes_index)
        self.index = self.hierarchy_prototypes_index[0]

    def make_hierarchy_hr_test(self, tau, min_samples=3):

        hyper_rectangles = list()
        if self.tau != tau:
            # print('1')
            self.tau = tau
        if self.min_samples != min_samples:
            self.min_samples = min_samples
        for index, x in zip(self.index, self.prototypes):

            hr = self.find_noise_test(x=x, index=index, min_samples=min_samples)
            if hr is not None:
                hyper_rectangles.append(hr)

        self.hyper_rectangles.appendleft(hyper_rectangles)

        self.get_prototypes_test()

    def fit_predict_test(self, tau, min_samples=3, stage=5):

        expected_proto = 500

        for step in range(stage):
            self.make_hierarchy_hr_test(tau)
            print(len(self.prototypes))
            # print(len(self.hyper_rectangles))
            print('expected number of prototype >= ', len(self.prototypes) / 2)
            expected_proto = len(self.prototypes) // 2

    def report_change_tau(self):
        tau = np.linspace(1e-3, 1, 100)

        DIR_PATH = os.getcwd() + r'\varying tau'

        for t in tqdm(tau):
            for i in trange(5):
                n_proto = []
                self.make_hierarchy_hr_test(tau=t)

                if not len(self.hyper_rectangles):
                    break
                print(len(self.hyper_rectangles[0]))
                n_proto.append(len(self.hyper_rectangles[0]))
                sleep(0.01)

                FILE_NAME = 'Hierarchy {}'.format(i + 1)
                PATH = os.path.join(DIR_PATH, FILE_NAME)
                plt.plot(tau, n_proto)
                plt.xlabel('tau', fontsize=15)
                plt.ylabel('Num. of prototypes', fontsize=15)
                plt.title('Num. of prototypes varying tau', fontsize=20)
                plt.savefig(PATH)

        # old fit_predict method code
        # hyper_rectangles = list()
        #
        # for index, x in zip(self.index, self.prototypes):
        #
        #     hr = self.find_noise(x=x, index=index, min_samples=min_samples)
        #     if hr is not None:
        #         hyper_rectangles.append(hr)
        #
        # self.hyper_rectangles.appendleft(hyper_rectangles)
        #
        # self.get_prototypes()
        #
        # self.tau = 0.095
        # hyper_rectangles = list()
        #
        # for index, x in zip(self.index, self.prototypes):
        #     if index in self.in_data:
        #         continue
        #     hr = self.find_noise(x=x, index=index, min_samples=min_samples*2)
        #     if hr is not None:
        #         hyper_rectangles.append(hr)
        #
        # self.hyper_rectangles.appendleft(hyper_rectangles)
        # self.get_prototypes()
        # self.tau = 0.08
        #
        # hyper_rectangles = list()
        #
        # for index, x in zip(self.index, self.prototypes):
        #     if index in self.in_data:
        #         continue
        #     hr = self.find_noise(x=x, index=index, min_samples=min_samples)
        #     if hr is not None:
        #         hyper_rectangles.append(hr)
        #
        # self.hyper_rectangles.appendleft(hyper_rectangles)
        # self.get_prototypes()
        # self.tau = 1
        #
        # hyper_rectangles = list()
        #
        # for index, x in zip(self.index,self.prototypes):
        #     if index in self.in_data:
        #         continue
        #     hr = self.find_noise(x=x, index=index, min_samples=min_samples)
        #     if hr is not None:
        #         hyper_rectangles.append(hr)
        # self.hyper_rectangles.appendleft(hyper_rectangles)

        # print(len(self.prototypes))
        # self.get_prototypes()
        # print(len(self.prototypes))
        # self.tau = self.tau * 2
        # print(self.index)
        # print(self.tau)
        #
        # hyper_rectangles = list()
        #
        # for index, x in zip(self.index,self.prototypes):
        #     if index in self.in_data:
        #         continue
        #     hr = self.find_noise(x=x, index=index, min_samples=min_samples)
        #     hyper_rectangles.append(hr)
        # self.hyper_rectangles.appendleft(hyper_rectangles)
        #
        # print(len(self.prototypes))
        # self.get_prototypes()
        # print(len(self.prototypes))
        # self.tau = self.tau * 2
        # print(self.index)
        # print(self.tau)
        #
        # hyper_rectangles = list()
        #
        # for index, x in zip(self.index, self.prototypes):
        #     if index in self.in_data:
        #         continue
        #     hr = self.find_noise(x=x, index=index, min_samples=min_samples)
        #     hyper_rectangles.append(hr)
        # self.hyper_rectangles.appendleft(hyper_rectangles)
        #
        # print(len(self.prototypes))
        # self.get_prototypes()
        # print(len(self.prototypes))
        # self.tau = self.tau * 2
        # print(self.index)
        # print(self.tau)

# print(len(self.prototypes))
# # print(len(self.hyper_rectangles))
# print('expected number of prototype >= ', len(self.prototypes) / 2)
# expected_proto = len(self.prototypes) // 2
#
# n_proto = []
#
# for sam in trange(min_samples, max_proto):
#     self.make_hierarchy_hr(tau=self.tau, min_samples=sam)
#     # print(len(self.prototypes))
#     n_proto.append(len(self.prototypes))
#     expected_proto = len(self.prototypes) // 2
#     n_proto_3 = []
#     for sam3 in trange(sam, max_proto, 10):
#         # print(sam3)
#         nproto = self.make_hierarchy_hr(tau=self.tau, min_samples=sam3, is_change=False)
#         n_proto_3.append(nproto)
#         sleep(0.01)
#     plt.plot(np.arange(len(n_proto_3)), n_proto_3)
#     plt.xticks(np.arange(0, len(n_proto_3)+1, 10), np.arange(0, len(n_proto_3)+1, 10), rotation=45)
#     plt.yticks(np.arange(0, len(n_proto_3)+1, 10), np.arange(0, len(n_proto_3)+1, 10))
#     plt.grid()
#     plt.show()
#
# self.make_hierarchy_hr(tau=0.045, min_samples=31)
# print('expected number of prototype >= ', len(self.prototypes) // 2)
# expected_proto = len(self.prototypes) // 2
#
# self.make_hierarchy_hr(tau=self.tau, min_samples=223)
# print(len(self.prototypes))
# # print(len(self.hyper_rectangles))
# print('expected number of prototype >= ', len(self.prototypes) // 2)
# expected_proto = len(self.prototypes) // 2