import numpy as np


class HyperRectangle:

    def __init__(self, x=None, x_index=None, y=None, tau=None, is_merge=False):

        self.hr_max = x + tau
        self.hr_min = x - tau
        self.hr_mid = x
        self.x_idx = x_index
        self.R = (self.hr_max - self.hr_min) / 2.0

        self.y = y

        self.covered_data_indexes = set([])
        if not is_merge:

            self.covered_data_indexes.add(x_index)

    def is_include(self, x_neighbor, index):

        is_in = True
        for new, mid, r in zip(x_neighbor, self.hr_mid,  self.R):
            if abs(new - mid) > r:
                is_in = False
                break

        if is_in:
            self.covered_data_indexes.add(index)
            return index, is_in
        else:
            return -1, is_in

    def __str__(self):
        text = '-'*50 + '\n hr max: {} \n hr min: {} \n hr mid: {} \n R: {}\n Covered indexes: {}\n'.format(self.hr_max, self.hr_min, self.hr_mid, self.R, self.covered_data_indexes) + '-'*50
        return text