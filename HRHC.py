from HR_TREE.Hyper_Rectangle import HyperRectangle
from copy import deepcopy
from collections import deque

import numpy as np


class HRHC:

    def __init__(self, X=None, tau=0.1, index=None):

        self.prototypes = X
        self.X = X
        self.tau = tau
        self.labels_ = None
        self.core_sample_indices_ = None
        self.hyper_rectangles = deque()
        self.in_data = set([])
        self.index = index
        self.hierarchy_prototypes = deque()
        self.hierarchy_prototypes_index = deque()




        # self.prototypes = list()
    # def init_hr_list(self):
    #
    #     for hr_id in self.prototypes:


    # def make_clusters(self):
    #
    #     cluster_index = 0
    #
    #     hr_list = deepcopy(self.hyper_rectangles)
    #
    #     hr_comp_list = list(combinations(hr_list, 2))
    #
    #     for hr in self.hyper_rectangles:
    #         if hr.y is not None:
    #             continue
    #
    #         hr.y = cluster_index
    #
    #         self._cluster_labeling(hr)
    #
    #         cluster_index += 1

    # def _cluster_labeling(self, hr):
    #
    #     for neighbor_index in hr.covered_data_indexes:
    #         hr_neighbor = self.hyper_rectangles[neighbor_index]
    #         if hr_neighbor.y == -1:
    #             hr_neighbor.y = hr.y
    #             continue
    #
    #         if hr_neighbor.y is None:
    #             hr_neighbor.y = hr.y
    #             self._cluster_labeling(hr_neighbor)

    def calc_statistic(self, hr1, hr2):
        index = list(hr1.covered_data_indexes)
        index.sort()
        hr1 = self.X[index]

        index = list(hr1.covered_data_indexes)
        index.sort()
        hr2 = self.X[index]

        hr1_max = np.max(hr1, axis=0)
        hr1_min = np.min(hr1, axis=0)
        hr1_mean = np.mean(hr1, axis=0)
        hr1_var = np.var(hr1, axis=0)

        hr1_statistic = [hr1_max, hr1_min, hr1_mean, hr1_var]

        hr2_max = np.max(hr2, axis=0)
        hr2_min = np.min(hr2, axis=0)
        hr2_mean = np.mean(hr2, axis=0)
        hr2_var = np.var(hr2, axis=0)

        hr2_statistic = [hr2_max, hr2_min, hr2_mean, hr2_var]

        return hr1_statistic, hr2_statistic

    def KLD(self, dist1, dist2, dx):
        return np.sum(np.where(dist1 != 0, dist1 * np.log(dist1 / dist2) * dx, 0))

    def find_noise(self, x, index, min_samples):

        hr = HyperRectangle(x=x, tau=self.tau, x_index=index)

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

    def get_prototypes(self):
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

    def fit_predict(self, min_samples=3):

        hyper_rectangles = list()

        for index, x in zip(self.index, self.prototypes):

            hr = self.find_noise(x=x, index=index, min_samples=min_samples)
            if hr is not None:
                hyper_rectangles.append(hr)

        self.hyper_rectangles.appendleft(hyper_rectangles)

        self.get_prototypes()

        self.tau = 0.095
        hyper_rectangles = list()

        for index, x in zip(self.index, self.prototypes):
            if index in self.in_data:
                continue
            hr = self.find_noise(x=x, index=index, min_samples=min_samples*2)
            if hr is not None:
                hyper_rectangles.append(hr)

        self.hyper_rectangles.appendleft(hyper_rectangles)
        self.get_prototypes()
        self.tau = 0.08

        hyper_rectangles = list()

        for index, x in zip(self.index, self.prototypes):
            if index in self.in_data:
                continue
            hr = self.find_noise(x=x, index=index, min_samples=min_samples)
            if hr is not None:
                hyper_rectangles.append(hr)

        self.hyper_rectangles.appendleft(hyper_rectangles)
        self.get_prototypes()
        self.tau = 1

        hyper_rectangles = list()

        for index, x in zip(self.index,self.prototypes):
            if index in self.in_data:
                continue
            hr = self.find_noise(x=x, index=index, min_samples=min_samples)
            if hr is not None:
                hyper_rectangles.append(hr)
        self.hyper_rectangles.appendleft(hyper_rectangles)

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

    def predict(self, X):

        n_samples = X.shape[0]
        y_new = np.ones(shape=n_samples, dtype=int) * -1

        for i, data in enumerate(X):
            new_hr = HyperRectangle(x=data, tau=self.tau, index=i)

            for j, hr in enumerate(self.hyper_rectangles):
                if hr.y == -1:
                    continue

                if new_hr.is_include(hr.hr_mid, None):
                    y_new[i] = hr.y

                    break

        return y_new