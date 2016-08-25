#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2015-2016, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import sys
import numpy as np
from hub_toolbox.Distances import cosine_distance as cos
from hub_toolbox.Distances import euclidean_distance as l2
#DEPRECATED
from hub_toolbox.Distances import Distance

def centering(X:np.ndarray, metric:str='vector', test_set_mask:np.ndarray=None):
    """
    Perform  centering, i.e. shift the origin to the data centroid.
    
    Centering of vector data X with n objects in an m-dimensional feature space.
    The mean of each feature is calculated and subtracted from each point [1].
    In distance based mode, it must be checked upstream, that the distance
    matrix is a gram matrix as described below!
    
    Parameters
    ----------
    X : ndarray
        - An (m x n) vector data matrix with n objects in an 
          m-dimensional feature space 
        - An (n x n) distance matrix of form K = X(X.T), if X is an (n x m) 
          matrix; and of form K = (X.T)X, if X is an (m x n) matrix, 
          where X.T denotes the transpose of X.
        
        NOTE: The type must be defined via parameter 'metric'!
        
    metric : {'vector', 'distance'}, optional (Default: 'vector')
        Define, whether 'X' is vector data or a distance matrix.
        
    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).
    
    Returns
    ------- 
    X_cent : ndarray
    
        - Centered vectors, when given vector data
        - Centered gram matrix, when given distance data.
        
    See also
    --------
    [1] Suzuki, I., Hara, K., Shimbo, M., Saerens, M., & Fukumizu, K. (2013). 
    Centering similarity measures to reduce hubs. In Proceedings of the 2013 
    Conference on Empirical Methods in Natural Language Processing (pp 613–623). 
    Retrieved from https://www.aclweb.org/anthology/D/D13/D13-1058.pdf
    """
    
    if metric == 'distance':
        if test_set_mask is not None:
            raise NotImplementedError("Distance based centering does not "
                                      "support train/test splits so far.")
        n = X.shape[0]
        H = np.identity(n) - (1.0/n) * np.ones((n, n))
        K = X # K = X.T.X must be provided upstream
        X_cent = H.dot(K).dot(H)
        return X_cent
    elif metric == 'vector':
        n = X.shape[0]
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
        else:
            train_set_mask = slice(0, n) #np.ones(n, np.bool)
        
        vectors_mean = np.mean(X[train_set_mask], 0)
        X_cent = X - vectors_mean
        return X_cent
    else:
        raise ValueError("Parameter 'metric' must be 'distance' or 'vector'.")

def weighted_centering(X:np.ndarray, metric:str='cosine', gamma:float=1., 
                       test_set_mask:np.ndarray=None):
    """
    Perform  weighted centering: shift origin to the weighted data mean
    
    Move the origin more actively towards hub objects in the dataset, 
    rather than towards the data centroid [2].
    
    Parameters
    ----------
    X : ndarray
        An (m x n) vector data matrix with n objects in an 
        m-dimensional feature space 
    
    metric : {'cosine', 'euclidean'}, optional (default: 'cosine')
        Distance measure used to place more weight on objects that are more 
        likely to become hubs. (Defined for 'cosine' in [2], 'euclidean' does 
        not make much sense and might be removed in the future).
    
    gamma : float, optional (default: 1.0)
        Controls how much we emphasize the weighting effect
        
        - gamma=0: equivalent to normal centering
        - gamma>0: move origin closer to objects with larger similarity 
          to other objects
    
    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).
    
    Returns
    ------- 
    X_wcent : ndarray
        Weighted centered vectors.
        
    See also
    --------
    [2] Suzuki, I., Hara, K., Shimbo, M., Saerens, M., & Fukumizu, K. (2013). 
    Centering similarity measures to reduce hubs. In Proceedings of the 2013 
    Conference on Empirical Methods in Natural Language Processing (pp 613–623). 
    Retrieved from https://www.aclweb.org/anthology/D/D13/D13-1058.pdf
    """
    n = X.shape[0]
                   
    # Indices of training examples
    if test_set_mask is not None:
        train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
    else:
        train_set_mask = slice(0, n)
    
    n_train = X[train_set_mask].shape[0]
    d = np.zeros(n)
    
    if metric == 'cosine':
        vectors_sum = X[train_set_mask].sum(0)
        for i in np.arange(n):
            d[i] = n_train * cos(np.array([X[i], vectors_sum/n_train]))[0, 1]
    # Using euclidean distances does not really make sense
    elif metric == 'euclidean':
        for i in range(n):
            displ_v = X[train_set_mask] - d[i]
            d[i] = np.sum(np.sqrt(displ_v * displ_v))
    else:
        raise ValueError("Weighted centering only supports cosine distances.")
    d_sum = np.sum(d ** gamma)
    w = (d ** gamma) / d_sum
    vectors_mean_weighted = np.sum(w.reshape(n, 1) * X, 0)
    X_wcent = X - vectors_mean_weighted
    return X_wcent

def localized_centering(X:np.ndarray, metric:str='cosine', kappa:int=40, 
                        gamma:float=1., test_set_mask:np.ndarray=None):
    """
    Perform localized centering.
    
    Reduce hubness in datasets according to the method proposed in [3].
    
    Parameters
    ----------
    X : ndarray
        An (m x n) vector data matrix with n objects in an 
        m-dimensional feature space 
        
    metric : {'cosine', 'euclidean'}
        Distance measure used to place more weight on objects that are more 
        likely to become hubs. (Defined for 'cosine' in [2], 'euclidean' does 
        not make much sense and might be removed in the future).
        
    kappa : int, optional (default: 40)
        Local segment size, determines the size of the local neighborhood for 
        calculating the local affinity. When kappa=n localized centering 
        reduces to standard centering.
        "select κ depending on the dataset, so that the correlation between
        Nk(x) and the local affinity <x, cκ(x)> is maximized" [3]
        
    gamma : float, optional (default: 1.0)
        Control the degree of penalty, so that used the similarity score 
        is smaller depending on how likely a point is to become a hub.
        "Parameter γ can be tuned so as to maximally reduce the skewness 
        of the Nk distribution" [3].
        
    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).
    
    Returns
    ------- 
    S_lcent : ndarray
        Secondary similarity (localized centering) matrix.
        
    See also
    --------
    [3] Hara, K., Suzuki, I., Shimbo, M., Kobayashi, K., Fukumizu, K., & 
    Radovanović, M. (2015). Localized centering: Reducing hubness in 
    large-sample data hubness in high-dimensional data. In AAAI ’15: 
    Proceedings of the 29th AAAI Conference on Artificial Intelligence 
    (pp. 2645–2651).
    """
    if test_set_mask is None:
        test_set_mask = np.zeros(X.shape[0], np.bool)
    
    if metric == 'cosine':
        # Rescale vectors to unit length
        v = X / np.sqrt((X ** 2).sum(-1))[..., np.newaxis]
        # for unit vectors it holds inner() == cosine()
        sim = 1 - cos(v)
    # Localized centering meaningful for Euclidean?
    elif metric == 'euclidean':
        v = X # no scaling here...
        sim = 1 / (1 + l2(v))
    else:
        raise ValueError("Localized centering only supports cosine distances.")
    
    n = sim.shape[0]
    local_affinity = np.zeros(n)
    for i in range(n):
        x = v[i]
        sim_i = sim[i, :].copy()
        # set similarity of test examples to zero to exclude them from fit
        sim_i[test_set_mask] = 0
        # also exclude self
        sim_i[i] = 0
        nn = np.argsort(sim_i)[::-1][1 : kappa+1]
        c_kappa_x = np.mean(v[nn], 0)
        if metric == 'cosine':
            # c_kappa_x has no unit length in general
            local_affinity[i] = np.inner(x, c_kappa_x)
            #local_affinity[i] = cosine(x, c_kappa_x)
        elif metric == 'euclidean':
            local_affinity[i] = 1 / (1 + np.linalg.norm(x-c_kappa_x))
        else:
            raise ValueError("Localized centering only "
                             "supports cosine distances.")
    sim_lcent = sim - (local_affinity ** gamma)
    return sim_lcent

def dis_sim_global(X:np.ndarray, test_set_mask:np.ndarray=None):
    """
    Calculate dissimilarity based on global 'sample-wise centrality' [4].
    
    Parameters
    ----------
    X : ndarray
        An (m x n) vector data matrix with n objects in an 
        m-dimensional feature space
          
    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).
        
    Returns
    -------
    D_dsg : ndarray
        Secondary distance (DisSimGlobal) matrix.
        
    See also
    --------
    [4] Hara, K., Suzuki, I., Kobayashi, K., Fukumizu, K., & 
    Radovanović, M. (2016). Flattening the density gradient for eliminating 
    spatial centrality to reduce hubness. Proceedings of the Thirtieth AAAI 
    Conference on Artificial Intelligence (AAAI ’16), 1659–1665. Retrieved from 
    http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12055/11787
    """
    
    n = X.shape[0]

    if test_set_mask is not None:
        train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
    else:
        train_set_mask = slice(0, n)
    
    c = X[train_set_mask].mean(0)
    xq_c = ((X - c) ** 2).sum(1)
    D_dsg = np.zeros((n, n))
    for x in range(n):
        for q in range(n):
            x_q = ((X[x, :] - X[q, :]) ** 2).sum()
            D_dsg[x, q] = x_q - xq_c[x] - xq_c[q]
    return D_dsg

def dis_sim_local(X:np.ndarray, k:int=10, test_set_mask:np.ndarray=None):
    """Calculate dissimilarity based on local 'sample-wise centrality' [5].
    
    Parameters
    ----------
    X : ndarray
        An (m x n) vector data matrix with n objects in an 
        m-dimensional feature space
          
    k : int, optional (default: 10)
        Neighborhood size used for determining the local centroids.
        Can be optimized as to maximally reduce hubness [5].
          
    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).
        
    Returns
    -------
    D_dsl : ndarray
        Secondary distance (DisSimLocal) matrix.
        
    See also
    --------
    [5] Hara, K., Suzuki, I., Kobayashi, K., Fukumizu, K., & 
    Radovanović, M. (2016). Flattening the density gradient for eliminating 
    spatial centrality to reduce hubness. Proceedings of the Thirtieth AAAI 
    Conference on Artificial Intelligence (AAAI ’16), 1659–1665. Retrieved from 
    http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12055/11787
    """
    
    n = X.shape[0]
    D = l2(X)
    # Exclude self distances from kNN lists:
    np.fill_diagonal(D, np.inf)
    c_k = np.zeros_like(X)
    
    if test_set_mask is not None:
        train_set_mask = np.setdiff1d(np.arange(n), test_set_mask)
        for i in range(n):
            knn_idx = np.argsort(D[i, train_set_mask])[0:k]
            c_k[i] = X[train_set_mask[knn_idx]].mean(0)
    else: # take all
        for i in range(n):
            knn_idx = np.argsort(D[i, :])[0:k]
            c_k[i] = X[knn_idx].mean(0)
    c_k_xy = ((X - c_k) ** 2).sum(1)
    disSim = np.zeros_like(D)
    for x in range(n):
        for y in range(x, n):
            x_y = ((X[x] - X[y]) ** 2).sum()
            disSim[x, y] = x_y - c_k_xy[x] - c_k_xy[y]
    return disSim + disSim.T - np.diag(np.diag(disSim))

###############################################################################
#
# DEPRECATED class
#
class Centering(object):
    """Transform data (in vector space) by various 'centering' approaches."""


    def __init__(self, vectors:np.ndarray=None, dist:np.ndarray=None, 
                 is_distance_matrix=False):
        """DEPRECATED"""
        if is_distance_matrix:
            self.distance_matrix = np.copy(dist)
            self.vectors = None
        else:
            self.distance_matrix = None
            self.vectors = np.copy(vectors)
                
    def centering(self, distance_based=False, test_set_mask=None):
        """DEPRECATED"""
        print("DEPRECATED: Please use Centering.centering() instead.", 
              file=sys.stderr)
        if self.vectors is not None:
            metric = 'vector'
            X = self.vectors
        elif distance_based:
            metric = 'distance'
            X = self.distance_matrix
        else:
            raise ValueError("No vectors given and distance_based not set.")
        return centering(X, metric, test_set_mask)
        
    def weighted_centering(self, gamma, 
                           distance_metric=Distance.cosine, test_set_mask=None):
        """DEPRECATED"""
        print("DEPRECATED: Please use Centering.weighted_centering() instead.", 
              file=sys.stderr)
        if distance_metric == Distance.cosine:
            metric = 'cosine'
        elif distance_metric == Distance.euclidean:
            metric = 'euclidean'
        else:
            raise ValueError("Unknown distance metric {}.".
                             format(distance_metric.__str__()))
        return weighted_centering(self.vectors, metric, gamma, test_set_mask)
    
    def localized_centering(self, kappa:int=20, gamma:float=1, 
        distance_metric=Distance.cosine, test_set_mask=None):
        """DEPRECATED"""
        print("DEPRECATED: Please use Centering.localized_centering() instead.", 
              file=sys.stderr)
        if distance_metric == Distance.cosine:
            metric = 'cosine'
        elif distance_metric == Distance.euclidean:
            metric = 'euclidean'
        else:
            raise ValueError("Unknown distance metric {}.".
                             format(distance_metric.__str__()))
        return localized_centering(self.vectors, metric, 
                                   kappa, gamma, test_set_mask)
        
    def dis_sim_global(self, test_set_mask=None):
        """DEPRECATED"""
        print("DEPRECATED: Please use Centering.disSim_glocal() instead.", 
              file=sys.stderr)
        return dis_sim_global(self.vectors, test_set_mask)
    
    def dis_sim_local(self, k, test_set_mask=None):
        """DEPRECATED"""
        print("DEPRECATED: Please use Centering.dis_sim_local() instead.", 
              file=sys.stderr)
        return dis_sim_local(self.vectors, k, test_set_mask)

if __name__ == '__main__':
    #vectors = np.arange(12).reshape(3,4)
    np.random.seed(47)
    VECT_DATA = np.random.rand(3, 4)
    print("Vectors: ............... \n{}".
          format(VECT_DATA))
    print("Centering: ............. \n{}".
          format(centering(VECT_DATA, 'vector')))
    print("Weighted centering: .... \n{}".
          format(weighted_centering(VECT_DATA, 'cosine', 0.4)))
    print("Localized centering: ... \n{}".
          format(localized_centering(VECT_DATA, 'cosine', 2, 1)))
    print("DisSim (global): ....... \n{}".
          format(dis_sim_global(VECT_DATA)))
    print("DisSim (local): ........ \n{}".
          format(dis_sim_local(VECT_DATA, k=2)))
