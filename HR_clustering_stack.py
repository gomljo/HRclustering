from HR_TREE.Hyper_Rectangle import HyperRectangle

import numpy as np


class HR_clustering:

    def __init__(self, X=None, tau=0.1):

        self.X_data = X
        self.tau = tau
        self.labels_ = None
        self.core_sample_indices_ = None
        self.hyper_rectangles = list()

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
                core_samples_indices.append(hr.x_idx)

        self.core_sample_indices_ = core_samples_indices
        return y_pred

    def make_clusters(self):
        stack = list()
        cluster_index = 0
        is_change_label = False
        self.hyper_rectangles = np.array(self.hyper_rectangles)

        for hr in self.hyper_rectangles:

            if hr.y is None:
                hr.y = cluster_index

            elif hr.y == -1:
                is_change_label = True
                continue

            neighbors = list(hr.covered_data_indexes)
            print(neighbors)
            stack.append(hr)
            hr_temp = stack.pop(-1)
            print(hr_temp)
            if len(list(hr_temp.covered_data_indexes)):
                stack.append(self.hyper_rectangles[list(hr_temp.covered_data_indexes)])
                print(stack)
                hr_1 = stack.pop(-1)
                print(hr_1[0])

            if is_change_label:
                is_change_label = False
                cluster_index += 1

    def _cluster_labeling(self, hr):

        for neighbor_index in hr.covered_data_indexes:
            hr_neighbor = self.hyper_rectangles[neighbor_index]
            if hr_neighbor.y == -1:
                hr_neighbor.y = hr.y
                continue

            if hr_neighbor.y is None:
                hr_neighbor.y = hr.y
                # self._cluster_labeling(hr_neighbor)

    def find_noise(self, min_samples):

        for index, x in enumerate(self.X_data):

            hr = HyperRectangle(x=x, tau=self.tau, x_index=index)

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
