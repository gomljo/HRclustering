from HR_TREE.Hyper_Rectangle import HyperRectangle

import numpy as np
from itertools import product, combinations
from copy import deepcopy

class HR_clustering:

    def __init__(self, X=None, tau=0.1):

        self.X_data = X
        self.tau = tau
        self.labels_ = None
        self.core_sample_indices_ = None
        self.hyper_rectangles = list()

    def calc(self, hr1, hr2):

        hr1_idx = list(hr1.covered_data_indexes)
        hr2_idx = list(hr2.covered_data_indexes)

        compare_list = list(product(hr1_idx, hr2_idx))
        dist_sum = 0

        for pair in compare_list:

            hr_dist = np.max(abs(self.X_data[pair[0]]-self.X_data[pair[1]]))

            dist_sum += hr_dist

        avg_dist = dist_sum / len(compare_list)

        return avg_dist

    def fit_predict(self, min_samples=3):

        self.find_noise(min_samples)
        self.make_clusters()

        y_pred = list()
        for hr in self.hyper_rectangles:
            y_pred.append(hr.y)

        y_pred = np.array(y_pred)
        self.labels_ = y_pred
        core_samples_indices = list()
        for hr in self.hyper_rectangles:
            if hr.y != -1:
                core_samples_indices.append(hr.hr_mid_index)

        self.core_sample_indices_ = core_samples_indices
        return y_pred

    def make_clusters(self):

        cluster_index = 0

        hr_list = deepcopy(self.hyper_rectangles)

        hr_comp_list = list(combinations(hr_list, 2))

        for hr in self.hyper_rectangles:
            if hr.y is not None:
                continue

            hr.y = cluster_index

            self._cluster_labeling(hr)

            cluster_index += 1

    def _cluster_labeling(self, hr):

        for neighbor_index in hr.covered_data_indexes:
            hr_neighbor = self.hyper_rectangles[neighbor_index]
            if hr_neighbor.y == -1:
                hr_neighbor.y = hr.y
                continue

            if hr_neighbor.y is None:
                hr_neighbor.y = hr.y
                self._cluster_labeling(hr_neighbor)

    def find_noise(self, min_samples):

        for index, x in enumerate(self.X_data):
            # print(x)
            hr = HyperRectangle(x=x, tau=self.tau, index=index)

            for idx, neighbor in enumerate(self.X_data):
                if index == idx:
                    continue

                hr.is_include(neighbor, idx)

            if len(hr.covered_data_indexes) < min_samples:
                hr.y = -1

            self.hyper_rectangles.append(hr)

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
