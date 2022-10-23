from Hyper_Rectangle import HyperRectangle

from collections import deque

import numpy as np


class HRHC:

    def __init__(self, X=None, tau=0.1, index=None, is_multi_var=False):

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

        self.is_multi_var = is_multi_var

    def get_hr_info(self, hr):

        index = list(hr.covered_data_indexes)
        index.sort()
        hr = self.X[index]

        hr_max = np.max(hr, axis=0)
        hr_min = np.min(hr, axis=0)
        hr_mean = np.mean(hr, axis=0)
        hr_var = np.var(hr, axis=0)

        return hr_max, hr_min, hr_mean, hr_var

    @staticmethod
    def get_density_range(x_min, x_max, num_of_dx=1000):
        x_range = np.linspace(x_min, x_max, num_of_dx)
        return x_range

    @staticmethod
    def get_prob_density(x_range, hr_var, hr_mean):
        return (1 / np.sqrt(2 * np.pi * hr_var)) * np.exp(-1 * ((x_range - hr_mean) ** 2 / (2 * hr_var)))

    @staticmethod
    def get_multi_variate_prob_density(x_range, hr_var, hr_mean):
        return (1 / np.sqrt(2 * np.pi * hr_var)) * np.exp(-1 * ((x_range - hr_mean) ** 2 / (2 * hr_var)))

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

    def get_info(self, hr1, hr2):
        """
        :param hr1: hyper rectangle hr_i in self.hyper_rectangles[0]
        :param hr2: hyper rectangle hr_j in self.hyper_rectangles[0]
        :return: y1: [num. of feature dimension vector], y2: [num. of feature dimension vector],
        dx: [num. of feature dimension vector],  x_min: [num. of feature dimension vector], x_max: [num. of feature dimension vector]
        """
        num_dx = 1000

        hr1_max, hr1_min, hr1_mean, hr1_var = self.get_hr_info(hr1)
        hr2_max, hr2_min, hr2_mean, hr2_var = self.get_hr_info(hr2)

        if hr2_max >= hr1_max:
            x_max = hr2_max
        else:
            x_max = hr1_max

        if hr2_min <= hr1_min:
            x_min = hr2_min
        else:
            x_min = hr1_min

        # get probability density need to range
        x_range = self.get_density_range(x_min, x_max, num_of_dx=num_dx)

        dx = (x_max - x_min) / num_dx

        y1 = self.get_prob_density(x_range, hr1_var, hr1_mean)
        y2 = self.get_prob_density(x_range, hr2_var, hr2_mean)

        return y1, y2, dx, x_min, x_max

    def KLD(self, dist1, dist2, dx):
        # calculate KL-Divergence between dist1 and dist2
        return np.sum(np.where(dist1 != 0, dist1 * np.log(dist1 / dist2) * dx, 0))

    def merging(self, x_min, x_max):
        """
        question on this process
        which point can be prototype for merging two hyper rectangle

        solution (1) assign to new hr's prototype to hr1's prototype (X)
        solution (2) assign to new hr's prototype to hr2's prototype (X)
        solution (3) assign to new hr's prototype to mean value of hr1's prototype and hr2's prototype (O)
        """
        new_prototype = (x_max - x_min) / 2.0
        new_tau = x_max - x_min
        new_hr = HyperRectangle(x=new_prototype, tau=new_tau)

        return new_hr

    def make_cluster(self):

        # this method makes sub cluster.
        # Each execution makes cluster hierarchy
        # and need to eliminate merging hyper rectangles. => use self.in_data

        # 병합을 위한 KLD 값의 순위를 구하려면 인덱스: KLD 형태의 딕셔너리 구조로 구현하는 것이 유리할 것 같다.

        for idx1, hr1 in enumerate(self.hyper_rectangles[0]):
            max_kld_exp = 0
            max_kld_idx = 0
            kld_dict = dict()
            for idx2, hr2 in enumerate(self.hyper_rectangles[0]):
                if idx1 == idx2:
                    continue
                # variable y is list type and represent normal distribution on min value and max value of two hyper
                # rectangles which hr1 and hr2
                # returned value is num. of feature dimension vector
                # so, you need to specify mode that is one variate norm distribution or multi variate norm distribution

                y1, y2, dx, min_x, max_x = self.get_info(hr1, hr2)

                kld_exp = np.exp(-self.KLD(y1, y2, dx))

                kld_dict[idx2] = kld_exp

                sort_kld_dict = sorted(kld_dict.items(), reverse=True)
                print(sort_kld_dict)

    def find_HR(self, x, index, min_samples):

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
        # hr.covered_data_indexes.add(self.hyper_rectangles[0][index].covered_data_indexes)
        return hr

    # def find_HR(self, x, index, min_samples):
    #
    #     hr = HyperRectangle(x=x, tau=self.tau, x_index=index)
    #
    #     for idx, neighbor in zip(self.index, self.prototypes):
    #         if index == idx:
    #             continue
    #         # elif idx in self.in_data:
    #         #     continue
    #         dup_ind = hr.is_include(neighbor, idx)
    #
    #     if len(hr.covered_data_indexes) < min_samples:
    #         hr.y = -1
    #         return None
    #     # hr.covered_data_indexes.add(self.hyper_rectangles[0][index].covered_data_indexes)
    #     return hr
    #
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

    def fit_predict(self, min_samples=3, ranking=2):

        hyper_rectangles = list()

        for index, x in zip(self.index, self.prototypes):

            hr = self.find_HR(x=x, index=index, min_samples=min_samples)
            if hr is not None:
                hyper_rectangles.append(hr)
        self.hyper_rectangles.appendleft(hyper_rectangles)

        self.make_cluster()

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
