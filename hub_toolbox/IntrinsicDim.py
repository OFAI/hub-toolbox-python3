#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2016, Dominik Schnitzer and Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>

This file is based on a Matlab script by Elizaveta Levina, University of
Michigan, available at http://dept.stat.lsa.umich.edu/~elevina/mledim.m

Reference:  E. Levina and P.J. Bickel (2005).
 "Maximum Likelihood Estimation  of Intrinsic Dimension."
 In Advances in NIPS 17, Eds. L. K. Saul, Y. Weiss, L. Bottou.
"""
import numpy as np

def intrinsic_dimension(X:np.ndarray, k1:int=6, k2:int=12,
                        estimator:str='mackay', metric:str='vector',
                        trafo:str=None, mem_threshold:int=5000):
    """Calculate intrinsic dimension based on the MLE by Levina and Bickel [1]_.

    Parameters
    ----------
    X : ndarray
        - An ``m x n`` vector data matrix with ``n`` objects in an
          ``m`` dimensional feature space
        - An ``n x n`` distance matrix.

        NOTE: The type must be defined via parameter `metric`!

    k1 : int, optional (default: 6)
        Start of neighborhood range to search in.

    k2 : int, optional (default: 12)
        End of neighborhood range to search in.

    estimator : {'levina', 'mackay'}, optional (default: 'mackay')
        Determine the summation strategy: see [2]_.

    metric : {'vector', 'distance'}, optional (default: 'vector')
        Determine data type of `X`.

        NOTE: the MLE was derived for euclidean distances. Using
        other dissimilarity measures may lead to undefined results.

    trafo : {None, 'std', 'var'}, optional (default: None)
        Transform vector data. 

        - None: no transformation
        - 'std': standardization
        - 'var': subtract mean, divide by variance (default behavior of
          Laurens van der Maaten's DR toolbox; most likely for other
          ID/DR techniques).

    mem_treshold : int, optional, default: 5000
        Controls speed-memory usage trade-off: If number of points is higher
        than the given value, don't calculate complete distance matrix at
        once (fast, high memory), but per row (slower, less memory).

    Returns
    -------
    d_mle : int
        Intrinsic dimension estimate (rounded to next integer)

    References
    ----------
    .. [1] Levina, E., & Bickel, P. (2004). Maximum likelihood estimation of
           intrinsic dimension. Advances in Neural Information …, 17, 777–784.
           http://doi.org/10.2307/2335172
    .. [2] http://www.inference.phy.cam.ac.uk/mackay/dimension/
    """
    n = X.shape[0]
    if estimator not in ['levina', 'mackay']:
        raise ValueError("Parameter 'estimator' must be 'levina' or 'mackay'.")
    if k1 < 1 or k2 < k1 or k2 >= n:
        raise ValueError("Invalid neighborhood: Please make sure that "
                         "0 < k1 <= k2 < n. (Got k1={} and k2={}).".
                         format(k1, k2))
    X = X.copy().astype(float)

    if metric == 'vector':
        # New array with unique rows
        X = X[np.lexsort(np.fliplr(X).T)]
        
        if trafo is None:
            pass
        elif trafo == 'var':
            X -= X.mean(axis=0) # broadcast
            X /= X.var(axis=0) + 1e-7 # broadcast
        elif trafo == 'std':
            # Standardization
            X -= X.mean(axis=0) # broadcast
            X /= X.std(axis=0) + 1e-7 # broadcast
        else:
            raise ValueError("Transformation must be None, 'std', or 'var'.")

        # Compute matrix of log nearest neighbor distances
        X2 = (X**2).sum(1)

        if n <= mem_threshold: # speed-memory trade-off
            distance = X2.reshape(-1, 1) + X2 - 2*np.dot(X, X.T) #2x br.cast
            distance.sort(1)
            # Replace invalid values with a small number
            distance[distance<0] = 1e-7
            knnmatrix = .5 * np.log(distance[:, 1:k2+1])
        else:
            knnmatrix = np.zeros((n, k2))
            for i in range(n):
                distance = np.sort(X2[i] + X2 - 2 * np.dot(X, X[i, :]))
                # Replace invalid values with a small number
                distance[distance < 0] = 1e-7
                knnmatrix[i, :] = .5 * np.log(distance[1:k2+1])
    elif metric == 'distance':
        raise NotImplementedError("ID currently only supports vector data.")
        #=======================================================================
        # # TODO calculation WRONG
        # X.sort(1)
        # X[X < 0] = 1e-7
        # knnmatrix = np.log(X[:, 1:k2+1])
        #=======================================================================
    elif metric == 'similarity':
        raise NotImplementedError("ID currently only supports vector data.")
        #=======================================================================
        # # TODO calculation WRONG
        # print("WARNING: using similarity data may return "
        #       "undefined results.", file=sys.stderr)
        # X[X < 0] = 0
        # distance = 1 - (X / X.max())
        # knnmatrix = np.log(distance[:, 1:k2+1])
        #=======================================================================
    else:
        raise ValueError("Parameter `metric` must be 'vector'.")

    # Compute the ML estimate
    S = np.cumsum(knnmatrix, 1)
    indexk = np.arange(k1, k2+1) # broadcasted afterwards
    dhat = -(indexk - 2) / (S[:, k1-1:k2] - knnmatrix[:, k1-1:k2] * indexk)
    if estimator == 'levina':
        # Average over estimates and over values of k
        no_dims = dhat.mean()
    if estimator == 'mackay':
        # Average over inverses
        dhat **= -1
        dhat_k = dhat.mean(0)
        no_dims = (dhat_k ** -1).mean()
    return int(no_dims.round())

if __name__ == '__main__':
    m_dim = 100
    n_dim = 2000
    VECT_DATA = np.random.rand(n_dim, m_dim)
    id_ = intrinsic_dimension(VECT_DATA)
    print("Random {}x{} matrix: ID_MLE = {}".format(n_dim, m_dim, id_))
