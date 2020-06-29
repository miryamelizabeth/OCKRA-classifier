# --------------------------------------------------------------------------------------------
# Created by Miryam Elizabeth Villa-Pérez
# 
# Source code based on the article:
# 
# J. Rodríguez, A. Barrera-Animas, L. Trejo, M. Medina-Pérez and R. Monroy, 'Ensemble of
# One-Class Classifiers for Personal Risk Detection Based on Wearable Sensor Data',
# Sensors 16(10), p. 1619, 2016, [online] Available: https://doi.org/10.3390/s16101619
# --------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_X_y


class OCKRA(BaseEstimator):
    
    def __init__(self, classifier_count=100, K=10, use_dist_threshold=False, user_threshold=95):
        
        self.classifier_count = classifier_count
        self.K = K
        self.use_dist_threshold = use_dist_threshold
        self.user_threshold = user_threshold

    
    def score_samples(self, X):
        
        X_test = pd.DataFrame(X)
        X_test = pd.DataFrame(self._scaler.transform(X_test[X_test.columns]), index=X_test.index, columns=X_test.columns)
        
        similarity = np.average([np.exp(-0.5 * np.power(np.amin(euclidean_distances(X_test[self._features_consider[i]], self._centers[i]), axis=1) / self._dist_threshold[i], 2)) for i in range(len(self._centers))], axis=0)
        
        return similarity


    def predict(self, X):

        if (len(X.shape) < 2):
            raise ValueError('Reshape your data')

        if (X.shape[1] != self.n_features_):
            raise ValueError('Reshape your data')

        if not self._is_threshold_Computed:
            
            x_pred_classif = self.score_samples(X)            
            x_pred_classif.sort()
            self._inner_threshold = x_pred_classif[(100 - self.user_threshold) * len(x_pred_classif) // 100]
            self._is_threshold_Computed = True
        
        y_pred_classif = self.score_samples(X)
        
        return [-1 if s <= self._inner_threshold else 1 for s in y_pred_classif]
    

    def fit(self, X, y):
        
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(X, y)
        
        self._is_threshold_Computed = False
        
        # Total of features in dataset
        self.n_features_ = X_train.shape[1]
        
        if self.n_features_ < 1:
            raise ValueError('Unable to instantiate the train dataset - Empty vector')
        
        self._scaler = MinMaxScaler()
        X_train = pd.DataFrame(X_train)
        X_train = pd.DataFrame(self._scaler.fit_transform(X_train[X_train.columns]), index=X_train.index, columns=X_train.columns)

        # Random features
        self._features_consider = [np.unique(np.random.choice(np.arange(self.n_features_), self.n_features_)) for x in range(self.classifier_count)]
        
        # Save centers clustering and threshold distance
        self._centers = []
        self._dist_threshold = np.empty(self.classifier_count)
        
        
        for i in range(self.classifier_count):
            
            projected_dataset = X_train[self._features_consider[i]]
            
            # Clustering K-means
            kmeans = KMeans(n_clusters=self.K, random_state=0).fit(projected_dataset)
            centers = kmeans.cluster_centers_
            self._centers.append(centers)
            
            # Distance threshold
            threshold = 1 if not self.use_dist_threshold else np.mean(euclidean_distances(projected_dataset.iloc[::60], projected_dataset.iloc[::60]))
            self._dist_threshold = np.insert(self._dist_threshold, i, threshold)
        
        
        return self
