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
        self.hierarchy_hyper_rectangles = deque()
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

    def get_info(self, hr1, hr2):
        index = list(hr1.covered_data_indexes)
        index.sort()
        hr1 = self.X[index]

        hr1_max = np.max(hr1, axis=0)
        hr1_min = np.min(hr1, axis=0)
        hr1_mean = np.mean(hr1, axis=0)
        hr1_var = np.var(hr1, axis=0)

        index = list(hr2.covered_data_indexes)
        index.sort()
        hr2 = self.X[index]

        hr2_max = np.max(hr2, axis=0)
        hr2_min = np.min(hr2, axis=0)
        hr2_mean = np.mean(hr2, axis=0)
        hr2_var = np.var(hr2, axis=0)

        if hr2_max >= hr1_max:
            x_max = hr2_max
        else:
            x_max = hr1_max

        if hr2_min <= hr1_min:
            x_min = hr2_min
        else:
            x_min = hr1_min

        num_dx = 1000

        x_range = np.linspace(x_min, x_max, num_dx)

        dx = (x_max - x_min) / num_dx

        y1 = (1 / np.sqrt(2 * np.pi * hr1_var)) * np.exp(-1 * ((x_range - hr1_mean) ** 2 / (2 * hr1_var)))
        y2 = (1 / np.sqrt(2 * np.pi * hr2_var)) * np.exp(-1 * ((x_range - hr2_mean) ** 2 / (2 * hr2_var)))

        return y1, y2, dx, x_min, x_max

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

    def merging(self, hr1, hr2):

        new_hr = HyperRectangle()


    def make_cluster(self):

        # this method makes sub cluster.
        # Each execution makes cluster hierarchy
        # and need to eliminate merging hyper rectangles. => use self.in_data
        for idx1, hr1 in enumerate(self.hyper_rectangles):
            max_kld_exp = 0
            max_kld_idx = 0
            for idx2, hr2 in enumerate(self.hyper_rectangles):
                if idx1 == idx2:
                    continue
                # variable y is list type and represent normal distribution on min value and max value of two hyper
                # rectangles which hr1 and hr2
                y1, y2, dx = self.get_info(hr1, hr2)

                kld_exp = np.exp(-self.KLD(y1, y2, dx))

                if max_kld_exp < kld_exp:
                    max_kld_exp = kld_exp
                    max_kld_idx = idx2

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
