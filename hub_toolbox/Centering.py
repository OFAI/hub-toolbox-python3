#!/usr/bin/env python3
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
from sklearn.metrics.pairwise import euclidean_distances
from hub_toolbox.Distances import cosine_distance as cos
from hub_toolbox import IO

def centering(X:np.ndarray, metric:str='vector', test_set_mask:np.ndarray=None):
    """
    Perform  centering, i.e. shift the origin to the data centroid.

    Centering of vector data `X` with ``n`` objects in an ``m``-dimensional
    feature space.
    The mean of each feature is calculated and subtracted from each
    point [1]_. In distance based mode, it must be checked upstream, that
    the distance matrix is a gram matrix as described below!

    Parameters
    ----------
    X : ndarray
        - An ``(n x m)`` vector data matrix with ``n`` objects in an
          ``m``-dimensional feature space
        - An ``(n x n)`` distance matrix
          of form ``K = X(X.T)``, if `X` is an ``(n x m)`` matrix;
          and of form ``K = (X.T)X``, if `X` is an ``(m x n)`` matrix,
          where ``X.T`` denotes the transpose of `X`.

        NOTE: The type must be defined via parameter 'metric'!

    metric : {'vector', 'inner'}, optional (Default: 'vector')
        Define, whether `X` is vector data or a Gram matrix of inner product
        similarities as described above.

    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).

    Returns
    ------- 
    X_cent : ndarray

        Centered vectors with shape (n, m), if given vector data.

    K_cent : ndarray

        Centered inner product similarities with shape (n, n), if given Gram matrix.
        
    References
    ----------
    .. [1] Suzuki, I., Hara, K., Shimbo, M., Saerens, M., & Fukumizu, K. (2013). 
           Centering similarity measures to reduce hubs. In Proceedings of the 
           2013 Conference on Empirical Methods in Natural Language Processing 
           (pp 613–623). 
           Retrieved from https://www.aclweb.org/anthology/D/D13/D13-1058.pdf
    """
    # Kernel based centering requires inner product similarities, NOT distances.
    # Since the parameter was previously erroneously called 'distance',
    # this is kept for compatibility reasons.
    if metric in ('similarity', 'distance', 'inner', 'inner_product'):
        if test_set_mask is not None:
            raise NotImplementedError("Kernel based centering does not "
                                      "support train/test splits so far.")
        IO._check_distance_matrix_shape(X)
        n = X.shape[0]
        H = np.identity(n) - np.ones((n, n)) / n
        # K = X.T.X must be provided upstream
        return H.dot(X).dot(H)
    elif metric == 'vector':
        n = X.shape[0]
        if test_set_mask is None:
            # center among all data
            return X - np.mean(X, 0)
        else: 
            # center among training data
            train_ind = np.setdiff1d(np.arange(n), test_set_mask)
            return X - np.mean(X[train_ind], 0)        
    else:
        raise ValueError("Parameter 'metric' must be 'inner' or 'vector'.")

def weighted_centering(X:np.ndarray, metric:str='cosine', gamma:float=1., 
                       test_set_mask:np.ndarray=None):
    """
    Perform  weighted centering: shift origin to the weighted data mean
    
    Move the origin more actively towards hub objects in the dataset, 
    rather than towards the data centroid [1]_.
    
    Parameters
    ----------
    X : ndarray
        An ``m x n`` vector data matrix with ``n`` objects in an 
        ``m`` dimensional feature space 
    
    metric : {'cosine', 'euclidean'}, optional (default: 'cosine')
        Distance measure used to place more weight on objects that are more 
        likely to become hubs. (Defined for 'cosine' in [1]_, 'euclidean' does 
        not make much sense and might be removed in the future).
    
    gamma : float, optional (default: 1.0)
        Controls how much we emphasize the weighting effect
        
        - ``gamma=0`` : equivalent to normal centering
        - ``gamma>0`` : move origin closer to objects with larger similarity 
          to other objects
    
    test_set_mask : ndarray, optional (default: None)
        Hold back data as a test set and perform centering on the remaining 
        data (training set).
    
    Returns
    ------- 
    X_wcent : ndarray
        Weighted centered vectors.
        
    References
    ----------
    .. [1] Suzuki, I., Hara, K., Shimbo, M., Saerens, M., & Fukumizu, K. (2013). 
           Centering similarity measures to reduce hubs. In Proceedings of the 
           2013 Conference on Empirical Methods in Natural Language Processing 
           (pp 613–623). 
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

def localized_centering(X:np.ndarray, Y:np.ndarray=None,
                        kappa:int=40, gamma:float=1.):
    """
    Perform localized centering.
    
    Reduce hubness in datasets according to the method proposed in [2]_.
    
    Parameters
    ----------
    X : ndarray
        An ``n x m`` vector data matrix with ``n`` objects in an 
        ``m`` dimensional feature space

    Y : ndarray, optional
        If Y is provided, calculate similarities between all test data in `X`
        versus all training data in `Y`.
    
    kappa : int, optional (default: 40)
        Local segment size, determines the size of the local neighborhood for 
        calculating the local affinity. When ``kappa=n`` localized centering 
        reduces to standard centering.
        "select κ depending on the dataset, so that the correlation between
        Nk(x) and the local affinity <x, cκ(x)> is maximized" [2]_
        
    gamma : float, optional (default: 1.0)
        Control the degree of penalty, so that used the similarity score 
        is smaller depending on how likely a point is to become a hub.
        "Parameter γ can be tuned so as to maximally reduce the skewness 
        of the Nk distribution" [2]_.

    Returns
    ------- 
    S_lcent : ndarray
        Secondary similarity (localized centering) matrix.
        
    References
    ----------
    .. [1] Suzuki, I., Hara, K., Shimbo, M., Saerens, M., & Fukumizu, K. (2013). 
           Centering similarity measures to reduce hubs. In Proceedings of the 
           2013 Conference on Empirical Methods in Natural Language Processing 
           (pp 613–623). 
           Retrieved from https://www.aclweb.org/anthology/D/D13/D13-1058.pdf
    
    .. [2] Hara, K., Suzuki, I., Shimbo, M., Kobayashi, K., Fukumizu, K., & 
           Radovanović, M. (2015). Localized centering: Reducing hubness in 
           large-sample data hubness in high-dimensional data. In AAAI ’15: 
           Proceedings of the 29th AAAI Conference on Artificial Intelligence 
           (pp. 2645–2651).
    """
    # Rescale vectors to unit length
    div_ = np.sqrt((X ** 2).sum(-1))[..., np.newaxis]
    div_[div_ == 0] = 1e-7
    v = X / div_
    if Y is None: # calc all-against-all in X
        w = v
        n, _ = X.shape
        sim = v.dot(w.T)
        sim_train = sim
    else: # calc sim from test data in X against train data in Y
        div_ = np.sqrt((Y ** 2).sum(-1))[..., np.newaxis]
        div_[div_ == 0] = 1e-7
        w = Y / div_
        n, _ = Y.shape
        sim = v.dot(w.T)
        sim_train = w.dot(w.T)   

    local_affinity = np.zeros(n)
    for i in range(n):
        # Get the kappa nearest neighbors (highest similarity)
        nn = np.argsort(sim_train[i, :])[-1:-kappa-1:-1]
        # Local centroid
        c_kappa_x = w[nn, :].mean(axis=0)
        local_affinity[i] = np.inner(w[i, :], c_kappa_x)
    # Only change penalty, if all values are positive 
    if gamma != 1 and (local_affinity < 0).sum() == 0:
        local_affinity **= gamma
    sim -= local_affinity
    return sim

def dis_sim_global(X:np.ndarray, Y:np.ndarray=None):
    """
    Calculate dissimilarity based on global 'sample-wise centrality' [1]_.
    
    Parameters
    ----------
    X : ndarray
        An ``n x m`` vector data matrix with ``n`` objects in an 
        ``m`` dimensional feature space

    Y : ndarray, optional
        If Y is provided, calculate dissimilarities between all test data
        in `X` and all training data in `Y`.

    Returns
    -------
    D_dsg : ndarray
        Secondary dissimilarity (DisSimGlobal) matrix.
        
    References
    ----------
    .. [1] Hara, K., Suzuki, I., Kobayashi, K., Fukumizu, K., & 
           Radovanović, M. (2016). Flattening the density gradient for 
           eliminating spatial centrality to reduce hubness. Proceedings of 
           the Thirtieth AAAI Conference on Artificial Intelligence (AAAI ’16), 
           1659–1665. Retrieved from http://www.aaai.org/ocs/index.php/AAAI/
           AAAI16/paper/download/12055/11787
    """
    if Y is None:
        Y = X
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have same number of features.")
    c = Y.mean(0)
    x_c = euclidean_distances(Y, c[np.newaxis, :], squared=True)
    if id(X) == id(Y): # i.e. no Y was provided
        q_c = x_c
    else: # avoid duplicate calculations
        q_c = euclidean_distances(X, c[np.newaxis, :], squared=True)
    D_xq = euclidean_distances(X, Y, squared=True)
    D_xq -= x_c.T
    D_xq -= q_c
    return D_xq

def dis_sim_local(X:np.ndarray, Y:np.ndarray=None, k:int=10):
    """Calculate dissimilarity based on local 'sample-wise centrality' [1]_.
    
    Parameters
    ----------
    X : ndarray
        An ``n x m`` vector data matrix with ``n`` objects in an 
        ``m`` dimensional feature space.

    Y : ndarray, optional
        If Y is provided, calculate dissimilarities between all test data
        in `X` and all training data in `Y`.

    k : int, optional (default: 10)
        Neighborhood size used for determining the local centroids.
        Can be optimized as to maximally reduce hubness [1]_.

    Returns
    -------
    D_dsl : ndarray
        Secondary dissimiliarity (DisSimLocal) matrix.
        
    References
    ----------
    .. [1] Hara, K., Suzuki, I., Kobayashi, K., Fukumizu, K., & 
           Radovanović, M. (2016). Flattening the density gradient for 
           eliminating spatial centrality to reduce hubness. Proceedings of 
           the Thirtieth AAAI Conference on Artificial Intelligence (AAAI ’16), 
           1659–1665. Retrieved from http://www.aaai.org/ocs/index.php/AAAI/
           AAAI16/paper/download/12055/11787
    """
    # all-against-all dissimilarities?
    if Y is None:
        Y = X

    # dataset size and dimensionality
    n_test, m_test = X.shape
    n_train, m_train = Y.shape
    if m_test != m_train:
        raise ValueError("X and Y must have same number of features.")

    # Calc euclidean distances to find nearest neighbors among training data
    D_train = euclidean_distances(Y, squared=True)
    if id(Y) == id(X):
        # Exclude self distances from kNN lists:
        np.fill_diagonal(D_train, np.inf)
        D_test = D_train
    else:
        # ... and between test and training data
        D_test = euclidean_distances(X, Y, squared=True)

    # Local centroid for each point among its k-nearest training neighbors
    c_k_X = np.zeros_like(X)
    for i in range(n_test):
        knn_idx = np.argsort(D_test[i, :])[:k]
        c_k_X[i] = Y[knn_idx].mean(axis=0)
    X -= c_k_X
    X **= 2
    x_c_k = X.sum(axis=1)
    if id(Y) == id(X):
        c_k_Y = c_k_X
        y_c_k = x_c_k
    else:
        c_k_Y = np.zeros_like(Y)
        for i in range(n_train):
            knn_idx = np.argsort(D_train[i, :])[:k]
            c_k_Y[i] = Y[knn_idx].mean(axis=0)
        Y -= c_k_Y
        Y **= 2
        y_c_k = Y.sum(axis=1)
    # DisSimLocal
    x_y = D_test
    x_y -= x_c_k[:, np.newaxis]
    x_y -= y_c_k
    if id(Y) == id(X):
        np.fill_diagonal(x_y, -np.inf)
    return x_y

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
          format(localized_centering(VECT_DATA, kappa=2, gamma=1)))
    print("DisSim (global): ....... \n{}".
          format(dis_sim_global(VECT_DATA)))
    print("DisSim (local): ........ \n{}".
          format(dis_sim_local(VECT_DATA, k=2)))
