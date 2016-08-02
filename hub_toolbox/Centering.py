#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2015-2016, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import numpy as np
from hub_toolbox.Distances import cosine_distance, Distance, euclidean_distance

class Centering(object):
    """Transform data (in vector space) by various 'centering' approaches."""


    def __init__(self, vectors:np.ndarray=None, dist:np.ndarray=None, 
                 is_distance_matrix=False):
        """Create an object for subsequent centering of vector data X with 
        n objects in an m-dimensional feature space.
        Set is_distance_matrix=True when using distance data.
        The distance matrix must be of form K = X(X.T), if X is an n x m matrix; 
        and of form K = (X.T)X, if X is an m x n matrix, where X.T denotes the
        transpose of X.
        
        """
        if is_distance_matrix:
            self.distance_matrix = np.copy(dist)
            self.vectors = None
        else:
            self.distance_matrix = None
            self.vectors = np.copy(vectors)
                
    def centering(self, distance_based=False, test_set_mask=None):
        """Perform standard centering, i.e. shift the origin to the data 
        centroid.
        
        The mean of each feature is calculated and subtracted from each point.
        
        In distance based mode, it must be checked upstream, that the distance
        matrix is a gram matrix as described in the constructor! 
        
        Returns centered vectors, when given vector data; and
        return centered gram matrix, when given distance data.
        """
            
        if self.distance_matrix is not None:
            if test_set_mask is not None:
                    raise NotImplementedError("Distance based centering does not "
                                              "support train/test splits so far.")
            n = self.distance_matrix.shape[0]
            H = np.identity(n) - (1.0/n) * np.ones((n, n))
            K = self.distance_matrix # K = X.T.X must be provided upstream
            K_cent = H.dot(K).dot(H)
            return K_cent
        else:
            n = self.vectors.shape[0]
            if test_set_mask is not None:
                train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
            else:
                train_set_mask = slice(0, n) #np.ones(n, np.bool)
            
            vectors_mean = np.mean(self.vectors[train_set_mask], 0)
            vectors_cent = self.vectors - vectors_mean
            return vectors_cent
        
    def weighted_centering(self, gamma, 
                           distance_metric=Distance.cosine, test_set_mask=None):
        """Perform weighted centering.
        
        Returns centered vectors (not distance matrix).
        """
        
        n = self.vectors.shape[0]
                   
        # Indices of training examples
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
        else:
            train_set_mask = slice(0, n) #np.ones(n, np.bool)
        
        n_train = self.vectors[train_set_mask].shape[0]
        d = np.zeros(n)
        
        if distance_metric == Distance.cosine:
            vectors_sum = self.vectors[train_set_mask].sum(0)
            for i in np.arange(n):
                #d[i] = n_train * np.inner(self.vectors[i], vectors_sum / n_train)
                #d[i] = n_train * cosine(self.vectors[i], vectors_sum / n_train)
                d[i] = n_train * cosine_distance(\
                        np.array([self.vectors[i], vectors_sum/n_train]))[0, 1]
        # Using euclidean distances does not really make sense
        elif distance_metric == Distance.euclidean:
            for i in range(n):
                displ_v = self.vectors[train_set_mask] - d[i]
                d[i] = np.sum(np.sqrt(displ_v * displ_v))
        else:
            raise ValueError("Weighted centering currently only supports "
                             "cosine distances.")
        d_sum = np.sum(d ** gamma)
        w = (d ** gamma) / d_sum
        vectors_mean_weighted = np.sum(w.reshape(n,1) * self.vectors, 0)
        vectors_weighted = self.vectors - vectors_mean_weighted
        return vectors_weighted
    
    def localized_centering(self, kappa:int=20, gamma:float=1, 
                        distance_metric=Distance.cosine, test_set_mask=None):
        """Perform localized centering.
        
        Returns a distance matrix (not centered vectors!)
        Default parameters: kappa=20, gamma=1.0
        """
        
        if test_set_mask is None:
            test_set_mask = np.zeros(self.vectors.shape[0], np.bool)
            
        if distance_metric == Distance.cosine:   
            # Rescale vectors to unit length
            v = self.vectors / np.sqrt((self.vectors ** 2).sum(-1))[..., np.newaxis]
            # for unit vectors it holds inner() == cosine()
            sim = 1 - cosine_distance(v)
        # Localized centering meaningful for Euclidean?
        elif distance_metric == Distance.euclidean:
            v = self.vectors # no scaling here...
            sim = 1 / ( 1 + euclidean_distance(v))
        else:
            raise ValueError("Localized centering currently only supports "
                             "cosine distances.")
        n = sim.shape[0]
        local_affinity = np.zeros(n)
        for i in range(n):
            x = v[i]
            sim_i = sim[i, :].copy()
            # set similarity of test examples to zero to exclude them from fit
            sim_i[test_set_mask] = 0
            # also exclude self 
            sim_i[i] = 0
            #TODO randomization?
            nn = np.argsort(sim_i)[::-1][1 : kappa+1]
            c_kappa_x = np.mean(v[nn], 0)
            if distance_metric == Distance.cosine:
                # c_kappa_x has no unit length in general
                local_affinity[i] = np.inner(x, c_kappa_x)       
                #local_affinity[i] = cosine(x, c_kappa_x) 
            elif distance_metric == Distance.euclidean:
                local_affinity[i] = 1 / (1 + np.linalg.norm(x-c_kappa_x))
            else:
                raise ValueError("Localized centering currently only supports "
                                 "cosine distances.")
        sim_lcent = sim - (local_affinity ** gamma)
        return 1 - sim_lcent
    
    def disSim_global(self, test_set_mask=None):
        """
        Calculate dissimilarity based on the notion of global 'sample-wise
        centrality'.
        
        This hubness reduction technique was proposed in Hara et al. (2016): 
        'Flattening the Density Gradient for Eliminating Spatial Centrality to 
        Reduce Hubness' for euclidean distances and isotropic Gaussian data 
        distributions.
        """
        
        n = self.vectors.shape[0]

        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
        else:
            train_set_mask = slice(0, n)#np.ones(self.vectors.shape[0], np.bool)
            
        c = self.vectors[train_set_mask].mean(0)
        xq_c = ((self.vectors - c) ** 2).sum(1)
        disSim = np.zeros((n, n))
        for x in range(n):
            for q in range(n):
                x_q = ((self.vectors[x, :] - self.vectors[q, :]) ** 2).sum()
                disSim[x, q] = x_q - xq_c[x] - xq_c[q]
        return disSim
    
    def disSim_local(self, k, test_set_mask=None):
        """
        Calculate dissimilarity based on the notion of local 'sample-wise 
        centrality'.
        
        This hubness reduction technique was proposed in Hara et al. (2016): 
        'Flattening the Density Gradient for Eliminating Spatial Centrality to 
        Reduce Hubness' for euclidean distances and non-isotropic Gaussian 
        data distributions.
        
        Parameter 'k' defines the neighborhood size used for determining the 
        local centroids.
        """
        
        n = self.vectors.shape[0]
        D = euclidean_distance(self.vectors)
        # Exclude self distances from kNN lists:
        np.fill_diagonal(D, np.inf) 
        c_k = np.zeros_like(self.vectors)
        
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
            for i in range(n):
                knn_idx = np.argsort(D[i, train_set_mask])[0:k]
                c_k[i] = self.vectors[train_set_mask[knn_idx]].mean(0)
        else: # take all    
            for i in range(n):
                knn_idx = np.argsort(D[i, :])[0:k]
                c_k[i] = self.vectors[knn_idx].mean(0)
        c_k_xy = ((self.vectors - c_k) ** 2).sum(1)
        disSim = np.zeros_like(D)
        for x in range(n):
            for y in range(x, n):
                x_y = ((self.vectors[x] - self.vectors[y]) ** 2).sum()
                disSim[x, y] = x_y - c_k_xy[x] - c_k_xy[y]
        return disSim + disSim.T - np.diag(np.diag(disSim))

if __name__ == '__main__':
    vectors = np.arange(12).reshape(3,4)
    np.random.seed(47)
    vectors = np.random.rand(3, 4)
    c = Centering(vectors)
    print("Vectors: ............... \n{}".format(vectors))
    print("Centering: ............. \n{}".format(c.centering()))
    print("Weighted centering: .... \n{}".format(c.weighted_centering(0.4)))
    print("Localized centering: ... \n{}".format(c.localized_centering(2)))
    print("DisSim (global): ....... \n{}".format(c.disSim_global()))
    print("DisSim (local): ........ \n{}".format(c.disSim_local(2)))