#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2016, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import numpy as np
from scipy.sparse.base import issparse
from hub_toolbox import Logging, IO

def score(D:np.ndarray, target:np.ndarray, k=5,
          metric:str='distance', test_set_ind:np.ndarray=None, verbose:int=0,
          sample_idx=None):
    """Perform `k`-nearest neighbor classification.

    Use the ``n x n`` symmetric distance matrix `D` and target class
    labels `target` to perform a `k`-NN experiment (leave-one-out
    cross-validation or evaluation of test set; see parameter `test_set_ind`).
    Ties are broken by the nearest neighbor.

    Parameters
    ----------
    D : ndarray
        The ``n x n`` symmetric distance (similarity) matrix.

    target : ndarray (of dtype=int)
        The ``n x 1`` target class labels (ground truth).

    k : int or array_like (of dtype=int), optional (default: 5)
        Neighborhood size for `k`-NN classification.
        For each value in `k`, one `k`-NN experiment is performed.

        HINT: Providing more than one value for `k` is a cheap means to perform
        multiple `k`-NN experiments at once. Try e.g. ``k=[1, 5, 20]``.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix

    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Perform a LOO-CV experiment
        - ndarray : Hold out points indexed in this array as test set. Fit
          model to remaining data. Evaluate model on test set.

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    Returns
    -------
    acc : ndarray (shape=(n_k x 1), dtype=float)
        Classification accuracy (`n_k`... number of items in parameter `k`)

        HINT: Refering to the above example... 
        ... ``acc[0]`` gives the accuracy of the ``k=1`` experiment.
    corr : ndarray (shape=(n_k x n), dtype=int)
        Raw vectors of correctly classified items

        HINT: ... ``corr[1, :]`` gives these items for the ``k=5`` experiment.
    cmat : ndarray (shape=(n_k x n_t x n_t), dtype=int) 
        Confusion matrix (``n_t`` number of unique items in parameter target)

        HINT: ... ``cmat[2, :, :]`` gives the confusion matrix of
        the ``k=20`` experiment.
    """

    # Check input sanity
    log = Logging.ConsoleLogging()
    if sample_idx is None:
        IO._check_distance_matrix_shape(D)
    else:
        IO._check_sample_shape_fits(D, sample_idx)
    IO._check_distance_matrix_shape_fits_labels(D, target)
    IO._check_valid_metric_parameter(metric)
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
    if metric == 'similarity':
        d_self = -np.inf
        sort_order = -1

    # Copy, because data is changed
    D = D.copy()
    target = target.astype(int)

    if verbose:
        log.message("Start k-NN experiment.")
    # Handle LOO-CV vs. test set mode
    if test_set_ind is None:
        n = D.shape[0]
        test_set_ind = range(n)    # dummy 
        train_set_ind = n   # dummy
    else:
        # number of points to be classified
        n = test_set_ind.size
        # Indices of training examples
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
        if sample_idx is not None:
            raise NotImplementedError("Sample k-NN does not support train/"
                                      "test splits at the moment.")
    # Number of k-NN parameters
    try:
        k_length = k.size
    except AttributeError as e:
        if isinstance(k, int):
            k = np.array([k])
            k_length = k.size
        elif isinstance(k, list):
            k = np.array(k)
            k_length = k.size
        else:
            raise e

    acc = np.zeros((k_length, 1))
    corr = np.zeros((k_length, D.shape[0]))

    cl = np.sort(np.unique(target))
    cmat = np.zeros((k_length, len(cl), len(cl)))

    classes = target.copy()
    for idx, cur_class in enumerate(cl):
        # change labels to 0, 1, ..., len(cl)-1
        classes[target == cur_class] = idx
    if sample_idx is not None:
        sample_classes = classes[sample_idx]
        j = np.ones(n, int)
        j *= (n+1) # illegal indices will throw index out of bounds error
        j[sample_idx] = np.arange(len(sample_idx))
        for j, sample in enumerate(sample_idx):
            D[sample, j] = d_self
    cl = range(len(cl))

    # Classify each point in test set
    for i in test_set_ind:
        seed_class = classes[i]

        if issparse(D):
            row = D.getrow(i).toarray().ravel()
        else:
            row = D[i, :]
        if sample_idx is None:
            row[i] = d_self

        # Sort points in training set according to distance
        # Randomize, in case there are several points of same distance
        # (this is especially relevant for SNN rescaling)
        if sample_idx is None:
            rp = train_set_ind
        else:
            rp = np.arange(len(sample_idx))
        rp = np.random.permutation(rp)
        d2 = row[rp]
        d2idx = np.argsort(d2, axis=0)[::sort_order]
        idx = rp[d2idx]

        # More than one k is useful for cheap multiple k-NN experiments at once
        for j in range(k_length):
            # Make sure no inf/-inf/nan values are used for classification
            finite_val = np.isfinite(row[idx[0:k[j]]])
            # However, if no values are finite, classify randomly
            if finite_val.sum() == 0:
                finite_val = np.ones_like(finite_val)
                log.warning("Query was classified randomly, because all "
                            "distances were non-finite numbers.")
            if sample_idx is None:
                nn_class = classes[idx[0:k[j]]][finite_val]
            else:
                #finite_val = np.isfinite(sample_row[idx[0:k[j]]])
                nn_class = sample_classes[idx[0:k[j]]][finite_val]
            cs = np.bincount(nn_class.astype(int))
            max_cs = np.where(cs == np.max(cs))[0]

            # "tie": use nearest neighbor
            if len(max_cs) > 1:
                if seed_class == nn_class[0]:
                    acc[j] += 1/n 
                    corr[j, i] = 1
                cmat[j, seed_class, nn_class[0]] += 1
            # majority vote
            else:
                if cl[max_cs[0]] == seed_class:
                    acc[j] += 1/n
                    corr[j, i] = 1
                cmat[j, seed_class, cl[max_cs[0]]] += 1

    if verbose:
        log.message("Finished k-NN experiment.")

    return acc, corr, cmat
