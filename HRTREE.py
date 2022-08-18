from Hyper_Rectangle import *
import matplotlib.pyplot as plt
import numpy as np


class HR_TREE:

    def __init__(self, X=None, tau=0.1, min_samples=3):

        self.tau = tau
        self.X = X
        self.hyper_rectangles = list()
        self.proper_hr_ind = np.zeros(2).astype('int64')
        self.min_samples = min_samples
        self.tree = list()

    def make_hr_candidates(self, min_samples):

        for index, x in enumerate(self.X):
            hr = HyperRectangle(x=x, x_index=index, tau=self.tau)

            for idx, neighbor in enumerate(self.X):
                if index == idx:
                    continue

                hr.is_include(neighbor, idx)

            if len(hr.covered_data_indexes) < min_samples:
                hr.y = -1

            self.hyper_rectangles.append(hr)

    def is_proper(self):

        proper_dist = -np.inf

        for standard_ind, hr in enumerate(self.hyper_rectangles):

            x1 = hr.hr_mid
            for compared_ind, hr2 in enumerate(self.hyper_rectangles):
                if standard_ind == compared_ind:
                    continue
                x2 = hr2.hr_mid
                max_dist = np.max(x1 - x2)
                if proper_dist < max_dist:
                    self.proper_hr_ind[0] = standard_ind
                    self.proper_hr_ind[1] = compared_ind
                    proper_dist = max_dist

    def calc_entropy(self, hr1, hr2):

        hr1_index = list(hr1.covered_data_indexes)
        hr2_index = list(hr2.covered_data_indexes)
        print(hr1_index)
        print(hr2_index)
        print(self.X[hr1_index])
        print(self.X[hr2_index])

        hr_merge_index = hr1_index + hr2_index
        # hr_merge_index = np.array(hr_merge_index)

        unique_val_1, counts_1 = (np.unique(hr1_index, return_counts=True))
        unique_val_2, counts_2 = (np.unique(hr2_index, return_counts=True))
        unique_val_merge, counts_merge = (np.unique(hr_merge_index, return_counts=True))
        print(np.unique(hr_merge_index, return_counts=True))
        prob_mass_1 = counts_1 / len(hr1_index)
        prob_mass_2 = counts_2 / len(hr2_index)
        prob_mass_merge = counts_merge / len(hr_merge_index)
        print(prob_mass_1)
        print(np.dot(prob_mass_1, np.log10(prob_mass_1)))
        print(np.sum(-1*prob_mass_1 * np.log10(prob_mass_1)))
        print(np.log10(prob_mass_1))
        hr1_entropy = -1 * (np.dot(prob_mass_1,  np.log10(prob_mass_1)))
        hr2_entropy = -1 * (np.dot(prob_mass_2, np.log10(prob_mass_2)))
        before_merge_entropy = (len(hr1_index) / len(hr_merge_index)) * hr1_entropy + (len(hr2_index) / len(hr_merge_index))*hr2_entropy
        hr_merge_entropy = -1 * (np.dot(prob_mass_merge, np.log10(prob_mass_merge)))

        return before_merge_entropy, hr_merge_entropy

    def merging(self):
        cnt = 0
        for ref_ind, hr1 in enumerate(self.hyper_rectangles):

            for test_ind, hr2 in enumerate(self.hyper_rectangles):
                if ref_ind == test_ind:
                    continue
                if hr1.y == -1 or hr2.y == -1:
                    continue
                before_ent, after_ent = self.calc_entropy(hr1, hr2)
                if before_ent > after_ent:
                    print(before_ent, after_ent)
                    cnt+=1
        print(cnt)

    def prop_find(self):

        proper_dist = -np.inf

        for standard_ind, hr in enumerate(self.hyper_rectangles):
            print(hr.covered_data_indexes)
            print(hr.hr_mid)
            print(self.X[list(hr.covered_data_indexes)])
            print(np.mean(self.X[list(hr.covered_data_indexes)]))
            print(np.var(self.X[list(hr.covered_data_indexes)]))
            for ind in hr.covered_data_indexes:
                print(self.hyper_rectangles[ind].hr_mid)
                print(self.hyper_rectangles[ind].covered_data_indexes)

                print(self.X[list(self.hyper_rectangles[ind].covered_data_indexes)])
                print(np.mean(self.X[list(self.hyper_rectangles[ind].covered_data_indexes)]))
                print(np.var(self.X[list(self.hyper_rectangles[ind].covered_data_indexes)]))

    def fit(self):

        self.make_hr_candidates(self.min_samples)
        # print('hr making is done')
        # print(self.hyper_rectangles[0])
        self.merging()
        # print(self.proper_hr_ind)
        # print(self.hyper_rectangles[self.proper_hr_ind[0]])
        # print(self.hyper_rectangles[self.proper_hr_ind[1]])
        # left_hr = self.hyper_rectangles[self.proper_hr_ind[0]]
        # right_hr = self.hyper_rectangles[self.proper_hr_ind[1]]
        # left_ind = self.hyper_rectangles[self.proper_hr_ind[0]].covered_data_indexes
        # right_ind = self.hyper_rectangles[self.proper_hr_ind[1]].covered_data_indexes
        # test = set(np.arange(0,150))
        # test -= left_ind
        # test -= right_ind
        # print(test)
        # left_x = self.X[list(left_ind)]
        # right_x = self.X[list(right_ind)]
        # print(left_x)
        # left_max_ind = np.argmax(left_x)
        # right_max_ind = np.argmax(right_x)
        #
        # plt.scatter(left_x[:,2], left_x[:,3], c='red')
        # plt.scatter(right_x[:,2], right_x[:,3], c='blue')
        #
        # plt.show()