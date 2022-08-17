from sklearn import metrics
from datetime import datetime
from LearnKit.AnomalyDetection.HyperRectangleDBSCAN import HyperRectangleClustering
from sklearn.cluster import DBSCAN

import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Analysis:

    def __init__(self, X, model_name, labels_true):
        self.X = X
        self.model = None
        self.model_name = model_name
        self.true = labels_true
        self.pred = None
        self.radius = None
        self.min_samples = None
        self.num_r = 0
        self.num_sam = 0

        self.core_samples_mask = None
        self.labels = None

        self.n_clusters_ = None
        self.n_noise_ = None
        self.n_core_samples = None
        self.homogeneity_score = None
        self.completeness_score = None
        self.v_measure = None
        self.adjusted_rand_index = None
        self.adjusted_mutual_information = None
        self.silhouette_coef = None

    def compute_metric(self):

        core_samples_mask = np.zeros_like(self.model.labels_, dtype=bool)
        core_samples_mask[self.model.core_sample_indices_] = True
        self.labels = self.model.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(self.pred)) - (1 if -1 in self.pred else 0)
        n_noise_ = list(self.pred).count(-1)
        n_core_samples = list(core_samples_mask).count(1)
        homogeneity_score = metrics.homogeneity_score(self.true, self.pred)
        completeness_score = metrics.completeness_score(self.true, self.pred)
        v_measure = metrics.v_measure_score(self.true, self.pred)
        adjusted_rand_index = metrics.adjusted_rand_score(self.true, self.pred)
        adjusted_mutual_information = metrics.adjusted_mutual_info_score(self.true, self.pred)
        silhouette_coef = metrics.silhouette_score(self.X, self.pred)

        return n_clusters_, n_noise_, n_core_samples, homogeneity_score, completeness_score, v_measure, adjusted_rand_index, adjusted_mutual_information, silhouette_coef

    def get_metrics(self, radius, min_samples):

        self.radius = radius
        self.min_samples = min_samples
        self.num_r = len(radius)
        self.num_sam = len(min_samples)

        self.n_clusters_ = np.zeros((self.num_r, self.num_sam))

        self.n_noise_ = np.zeros((self.num_r, self.num_sam))
        self.n_core_samples = np.zeros((self.num_r, self.num_sam))
        self.homogeneity_score = np.zeros((self.num_r, self.num_sam))
        self.completeness_score = np.zeros((self.num_r, self.num_sam))
        self.v_measure = np.zeros((self.num_r, self.num_sam))
        self.adjusted_rand_index = np.zeros((self.num_r, self.num_sam))
        self.adjusted_mutual_information = np.zeros((self.num_r, self.num_sam))
        self.silhouette_coef = np.zeros((self.num_r, self.num_sam))
        print(self.model_name+' is progressed')
        for idx, eps in enumerate(tqdm(self.radius)):
            time.sleep(0.01)
            for idx2, min_sam in enumerate(self.min_samples):
                if self.model_name == 'dbscan':
                    self.model = DBSCAN(eps=eps, min_samples=min_sam)
                    y_pred = self.model.fit_predict(self.X)
                elif self.model_name == 'hrdbscan':
                    self.model = HyperRectangleClustering(X=self.X, tau=eps)
                    y_pred = self.model.fit_predict(min_sam)

                self.pred = y_pred

                if len(set(y_pred)) <= 1:
                    continue
                n_clusters_, n_noise_, n_core_samples, homogeneity_score, completeness_score, v_measure, adjusted_rand_index, adjusted_mutual_information, silhouette_coef = self.compute_metric()
                self.n_clusters_[idx, idx2] = n_clusters_
                self.n_noise_[idx, idx2] = n_noise_
                self.n_core_samples[idx, idx2] = n_core_samples
                self.homogeneity_score[idx, idx2] = homogeneity_score
                self.completeness_score[idx, idx2] = completeness_score
                self.v_measure[idx, idx2] = adjusted_rand_index
                self.adjusted_rand_index[idx, idx2] = v_measure
                self.adjusted_mutual_information[idx, idx2] = adjusted_mutual_information
                self.silhouette_coef[idx, idx2] = silhouette_coef
