import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from copy import deepcopy


class Hyper_rectangle:

    def __init__(self, initial_prototype_idx, center, diameter=None):

        self.center = center
        self.x_max = center
        self.x_min = center
        self.index = initial_prototype_idx
        self.diameter = diameter
        self.cover_index = set()
        self.cover_index.add(self.index)
        self.y = None
        self.is_merging = False

    def update(self, neighbor, neighbor_idx):

        assert (neighbor.shape[0] > 0), 'There are no neighbor points. Please check out variable neighbor'
        assert (len(neighbor_idx) > 0), 'There are no neighbor points. Please check out variable neighbor_idx'

        # for pts, pts_idx in zip(neighbor, neighbor_idx):
        #
        #     self.x_min = np.minimum(self.x_min, pts)
        #     self.x_max = np.maximum(self.x_max, pts)
        #     self.diameter = np.max((self.x_max - self.x_min) / 2.0)
        #     self.center = (self.x_max + self.x_min) / 2.0
        #     self.cover_index.add(pts_idx)

        x_min = np.vstack([self.x_min, neighbor])
        x_max = np.vstack([self.x_max, neighbor])
        # print(x_min)
        # print(np.min(x_min, axis=0))

        self.x_min = np.min(x_min, axis=0)
        self.x_max = np.max(x_max, axis=0)
        self.diameter = np.max((self.x_max - self.x_min) / 2.0)
        self.center = (self.x_max + self.x_min) / 2.0
        self.cover_index.update(neighbor_idx)

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = '----------- Overview on a rectangle -----------\n'
        msg += 'index: {0}, min: {1}, max: {2}\n'.format(self.index, self.x_min, self.x_max)
        msg += 'middle: {0}, R: {1}\n'.format(self.center, self.diameter)
        msg += 'covered data set: {0}\n'.format(self.cover_index)
        msg += '------------------------------------------------'

        return msg


class Adaptive_hyper_rectangle:

    def __init__(self, min_pts=None, debug=False):
        self.X = None
        self.min_pts = min_pts
        self.debug = debug
        self.dist_matrix = None
        self.prototypes = None
        self.label_ = None

        self.hyper_rectangles = list()
        self.visit_mask = None
        self.new_hyper_rectangles_mask = None
        self.clusters = list()

    def init_params(self, X):
        self.X = X
        self.dist_matrix = np.zeros((self.X.shape[0], self.X.shape[0]))
        self.visit_mask = np.zeros(self.X.shape[0])
        np.fill_diagonal(self.dist_matrix, 0)

    def make_rectangle(self):

        from itertools import combinations
        n_data = 0
        comb = np.arange(self.X.shape[1])

        comb = list(combinations(comb, 2))

        for ch, c in enumerate(comb):
            if len(comb)-1:
                plt.subplot(1, len(comb), ch + 1)

            for idx, hr in enumerate(self.hyper_rectangles):
                n_data += len(hr.cover_index)

                plt.scatter(self.X[list(hr.cover_index), c[0]], self.X[list(hr.cover_index), c[1]], s=40)

                plt.vlines(x=hr.center[c[0]] + (hr.diameter), ymin=hr.center[c[1]] - (hr.diameter),
                           ymax=hr.center[c[1]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])
                plt.vlines(x=hr.center[c[0]] - (hr.diameter), ymin=hr.center[c[1]] - (hr.diameter),
                           ymax=hr.center[c[1]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])
                plt.hlines(y=hr.center[c[1]] + (hr.diameter), xmin=hr.center[c[0]] - (hr.diameter),
                           xmax=hr.center[c[0]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])
                plt.hlines(y=hr.center[c[1]] - (hr.diameter), xmin=hr.center[c[0]] - (hr.diameter),
                           xmax=hr.center[c[0]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])

            plt.scatter(self.X[:, c[0]], self.X[:, c[1]], c='black', marker='1', label='original')

            plt.legend()
        plt.show()

    def visualize(self):
        n_data = 0
        from itertools import combinations

        comb = np.arange(self.X.shape[1])

        comb = list(combinations(comb, 2))

        for ch, c in enumerate(comb):
            if len(comb) > 1:
                plt.subplot(1, len(comb), ch+1)

            for idx, hr in enumerate(self.clusters):

                n_data += len(hr.cover_index)

                plt.scatter(self.X[list(hr.cover_index), c[0]], self.X[list(hr.cover_index), c[1]], marker='+')

                plt.vlines(hr.center[c[0]] + (hr.diameter), hr.center[c[1]] - (hr.diameter), hr.center[c[1]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])
                plt.vlines(hr.center[c[0]] - (hr.diameter), hr.center[c[1]] - (hr.diameter), hr.center[c[1]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])
                plt.hlines(hr.center[c[1]] + (hr.diameter), hr.center[c[0]] - (hr.diameter), hr.center[c[0]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])
                plt.hlines(hr.center[c[1]] - (hr.diameter), hr.center[c[0]] - (hr.diameter), hr.center[c[0]] + (hr.diameter),
                           colors=list(mcolors.XKCD_COLORS)[idx])

            plt.scatter(self.X[:, c[0]], self.X[:, c[1]], c='black', marker='1', label='original')
            # print(n_data)
            plt.legend()
        plt.show()

    def calc_hr_dist_matrix(self):

        for idx in range(len(self.X)):

            for idx2 in range(idx+1, len(self.X)):

                self.dist_matrix[idx, idx2] = np.max(np.abs(self.X[idx] - self.X[idx2]))
        self.dist_matrix += self.dist_matrix.T

        if self.debug:
            print('Calculated distance matrix')
            print(self.dist_matrix)

    def make_hr(self):

        sort_idx = np.argsort(self.dist_matrix, axis=1)

        for i in range(sort_idx.shape[0]):
            if self.min_pts == 2:

                hr = Hyper_rectangle(i, self.X[i, :])
                neighbor_idx = list()

                neighbor_idx.append(sort_idx[i][self.min_pts-1])
                hr.update(self.X[sort_idx[i][self.min_pts-1]], neighbor_idx)
                self.hyper_rectangles.append(hr)

            else:
                hr = Hyper_rectangle(i, self.X[i, :])
                hr.update(self.X[sort_idx[i][1:self.min_pts]], sort_idx[i][1:self.min_pts])
                self.hyper_rectangles.append(hr)

        if self.debug:
            print('Num. of Hyper Rectangle: ', len(self.hyper_rectangles))
            for hr in self.hyper_rectangles:
                print(hr)

    def is_include(self, hyper_rectangle, x):

        distance = max(abs(hyper_rectangle.center - x))

        is_in = (distance <= hyper_rectangle.diameter)
        if self.debug:
            print('info')
            print('x: {}, hr: {}'.format(x, hyper_rectangle.center))
            print('x_max: ', hyper_rectangle.x_max)
            print('x_min: ', hyper_rectangle.x_min)
            print('hr\'s center point: ', hyper_rectangle.center)
            print('distance: ', distance)
            print('diameter: ', hyper_rectangle.diameter)

        return is_in

    def merge_sort(self, hr_list):
        hyper_rectangles = deepcopy(hr_list)
        n = len(hyper_rectangles)

        if n <= 1:
            return

        mid = n // 2
        left_group = hyper_rectangles[:mid]
        right_group = hyper_rectangles[mid:]

        self.merge_sort(left_group)
        self.merge_sort(right_group)

        left = 0
        right = 0
        now = 0

        while left < len(left_group) and right < len(right_group):
            if left_group[left].diameter < right_group[right].diameter:
                hyper_rectangles[now] = left_group[left]
                left += 1
                now += 1
            else:
                hyper_rectangles[now] = right_group[right]
                right += 1
                now += 1

        while left < len(left_group):
            hyper_rectangles[now] = left_group[left]
            left += 1
            now += 1

        while right < len(right_group):
            hyper_rectangles[now] = right_group[right]
            right += 1
            now += 1
        self.hyper_rectangles = hyper_rectangles

    def merging(self, hr1, hr2):
        if self.debug:
            if len(hr1.cover_index.intersection(hr2.cover_index)):
                print('hyper rectangle\'s {}th hr and {}th hr intersection'.format(self.hyper_rectangles.index(hr1),
                                                                                   self.hyper_rectangles.index(hr2)))

                print('hr1\'s cover index {}, hr2\'s cover index {}'.format(hr1.cover_index, hr2.cover_index))
                inter = hr1.cover_index.intersection(hr2.cover_index)

        is_in = False

        if hr1.cover_index.intersection(hr2.cover_index):
            is_in = True
            hr2.is_merging = True
            hr1.update(self.X[list(hr2.cover_index)], hr2.cover_index)

        # for hr2_x in self.X[list(hr2.cover_index)]:
        #     is_in = self.is_include(hr1, hr2_x)
        #     if is_in:
        #         hr2.is_merging = True
        #         hr1.update(self.X[list(hr2.cover_index)], hr2.cover_index)
        #         return is_in

        return is_in

    def make_cluster(self):
        # dup_idx = set()

        self.new_hyper_rectangles_mask = np.zeros(len(self.hyper_rectangles))
        is_change_total = True

        while is_change_total:
            change_cnt = 0
            self.clusters = list()
            for idx1, hr1 in enumerate(self.hyper_rectangles):
                # new_hr = None

                is_in = False
                if hr1.is_merging:
                    continue

                for idx2, hr2 in enumerate(self.hyper_rectangles):
                    #
                    if hr2.is_merging or (idx1 == idx2):
                        continue
                    is_in = self.merging(hr1, hr2)
                    if is_in:
                        change_cnt += 1

                new_hr = deepcopy(hr1)

                self.clusters.append(new_hr)
                # print(new_hr.is_merging)
                hr1.is_merging = True

            # print(change_cnt)
            # print(len(self.clusters))
            # if len(self.clusters) <= 33:
            #     is_change_total = False
            # # self.visualize()
            # else:
            self.hyper_rectangles = deepcopy(self.clusters)
            if not change_cnt:
                # print('change cnt is smaller than 1: ', change_cnt)
                is_change_total = False

    def make_cluster_ver2(self):
        # dup_idx = set()

        # self.visit_mask = np.zeros(self.X.shape[0]))

        for idx1, hr1 in enumerate(self.hyper_rectangles):

            if hr1.is_merging or self.visit_mask[idx1]:
                continue

            for idx2, hr2 in enumerate(self.hyper_rectangles):
                # print(hr2.is_merging)
                if hr2.is_merging or (idx1 == idx2) or self.visit_mask[idx2]:
                    continue
                is_in = self.merging(hr1, hr2)

            self.visit_mask[list(hr1.cover_index)] = 1
            # print(self.visit_mask)
            new_hr = deepcopy(hr1)
            # print(hr1.cover_index)
            self.clusters.append(new_hr)
            # self.new_hyper_rectangles_mask[idx1] = 1
            hr1.is_merging = True

            # self.visualize()

    def make_cluster_ver3(self):
        # dup_idx = set()

        # self.visit_mask = np.zeros(self.X.shape[0]))
        self.merge_sort(self.hyper_rectangles)
        for idx1, hr1 in enumerate(self.hyper_rectangles):
            # or self.visit_mask[idx1]
            if hr1.is_merging :
                continue

            for idx2, hr2 in enumerate(self.hyper_rectangles):
                # print(hr2.is_merging)
                # or self.visit_mask[idx2]
                if hr2.is_merging or (idx1 == idx2) :
                    continue
                is_in = self.merging(hr1, hr2)

            self.visit_mask[list(hr1.cover_index)] = 1
            # print(self.visit_mask)
            new_hr = deepcopy(hr1)
            # print(hr1.cover_index)
            self.clusters.append(new_hr)
            # self.new_hyper_rectangles_mask[idx1] = 1
            hr1.is_merging = True

    #
    # def find_missing(self):
    #     new_cluster = list()
    #     for idx1, hr1 in enumerate(self.clusters):
    #
    #         for idx2, hr2 in enumerate(self.clusters):
    #
    #             if idx1 == idx2:
    #                 continue
    #             if hr1.cover_index.intersection(hr2.cover_index):
    #                 hr2.is_merging = True
    #                 hr1.update(self.X[list(hr2.cover_index)], hr2.cover_index)
    #
    #         new_cluster.append(deepcopy(hr1))

    def labeling(self):

        self.label_ = np.zeros(self.X.shape[0])
        # print(len(self.clusters))
        for label, hr in enumerate(self.clusters):
            # print(hr.cover_index)
            hr.y = label
            self.label_[list(hr.cover_index)] = label

    def fit_predict(self, X):

        self.init_params(X)
        self.calc_hr_dist_matrix()
        self.make_hr()
        if self.debug:
            self.make_rectangle()
        self.make_cluster_ver3()
        self.labeling()

        return self.label_

    def cluster_info(self):

        for idx, cl in enumerate(self.clusters):
            print('{}th cluster\'s cover index: ', cl.cover_index)
            print('{}th cluster\'s diameter: ', cl.diameter)
            print('{}th cluster\'s center: ', cl.center)