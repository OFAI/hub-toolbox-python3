#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2017, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import ctypes
import multiprocessing as mp
import numpy as np
from scipy.sparse.base import issparse
from hub_toolbox import Logging, IO
from sklearn.preprocessing.label import LabelEncoder
from scipy.sparse.csr import csr_matrix
from functools import partial

__all__ = ['score', 'predict', 'r_precision',
           'f1_score', 'f1_macro', 'f1_micro', 'f1_weighted']

def score(D:np.ndarray, target:np.ndarray, k=5,
          metric:str='distance', test_set_ind:np.ndarray=None, verbose:int=0,
          sample_idx=None, filter_self=True):
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

    sample_idx : ...
        TODO add description

    filter_self : bool, optional, default: True
        Remove self similarities from sparse ``D``.
        This assumes that the highest similarity per row is the self
        similarity.
        
        NOTE: Quadratic dense matrices are always filtered for self
        distances/similarities, even if `filter_self` is set t0 `False`.
        
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
        IO.check_distance_matrix_shape(D)
    else:
        IO.check_sample_shape_fits(D, sample_idx)
    IO.check_distance_matrix_shape_fits_labels(D, target)
    IO.check_valid_metric_parameter(metric)
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
    if metric == 'similarity':
        d_self = -np.inf
        sort_order = -1

    # Copy, because data is changed
    D = D.copy()
    target = target.astype(int)
    D_is_sparse = issparse(D)

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
    if D_is_sparse:
        # Add a label for unknown class (object w/o nonzero sim to any others)
        cl = np.append(cl, cl.max()+1)
        n_classes = len(cl) + 1
    else:
        n_classes = len(cl)
    cmat = np.zeros((k_length, n_classes, n_classes))

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

        if D_is_sparse:
            row = D.getrow(i)
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
        if D_is_sparse:
            nnz = row.nnz
            rp = np.random.permutation(nnz)
            d2 = row.data[rp]
            # Partition for each k value
            kth = nnz - k - 1
            # sort the two highest similarities to end
            kth = np.append(kth, [nnz-2, nnz-1])
            # Clip negative indices (nnz < k)
            np.clip(kth, a_min=0, a_max=nnz-1, out=kth)
            # Remove duplicate k values and sort
            kth = np.unique(kth)
            d2idx = np.argpartition(d2, kth=kth)
            d2idx = d2idx[~np.isnan(d2[d2idx])][::-1]
            idx = row.nonzero()[1][rp[d2idx]]
            idx = idx[1:] # rem self sim
        else:
            rp = np.random.permutation(rp)
            d2 = row[rp]
            d2idx = np.argsort(d2, axis=0)[::sort_order]
            d2idx = d2idx[~np.isnan(d2[d2idx])] # filter NaN values
            idx = rp[d2idx]

        # More than one k is useful for cheap multiple k-NN experiments at once
        for j in range(k_length):
            # Make sure no inf/-inf/nan values are used for classification
            if D_is_sparse:
                #print(row[0, idx[0:k[j]]].toarray())
                finite_val = np.isfinite(row[0, idx[0:k[j]]].toarray().ravel())
                #print(finite_val)
            else:
                finite_val = np.isfinite(row[idx[0:k[j]]])
            # However, if no values are finite, classify randomly
            if finite_val.sum() == 0:
                idx = np.random.permutation(idx)
                finite_val = np.ones_like(finite_val)
                log.warning("Query was classified randomly, because all "
                            "distances were non-finite numbers.")
            if sample_idx is None:
                nn_class = classes[idx[0:k[j]]][finite_val]
            else:
                #finite_val = np.isfinite(sample_row[idx[0:k[j]]])
                nn_class = sample_classes[idx[0:k[j]]][finite_val]
            cs = np.bincount(nn_class.astype(int))
            if cs.size > 0:
                max_cs = np.where(cs == np.max(cs))[0]
            else:
                max_cs = np.array([len(cl) - 1]) # misclassification label
            #===================================================================
            # except:
            #     print("nnz:", nnz)
            #     print("rp:", rp)
            #     print("d2:", d2)
            #     print("kth:", kth)
            #     print("d2idx", d2idx)
            #     print("idx", idx)
            #     print("seed_class:", seed_class)
            #     print("cs", cs)
            #     print("nnclass:", nn_class)
            #     print("j:", j)
            #     print("k[j]:", k[j])
            #     print("idx[:k[j]]:", idx[:k[j]])
            #     print("finite_val:", finite_val)
            #     print("Classes[:20]:", classes[:20])
            #     print("row:", row)
            #     print("row[0]:", row[0])
            #     print("row.shape:", row.shape, row[0].shape)
            #     return
            #===================================================================

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

def predict(D:np.ndarray, target:np.ndarray, k=5,
            metric:str='distance', test_ind:np.ndarray=None, verbose:int=0,
            sample_idx=None, return_cmat=True):
    """Perform `k`-nearest neighbor classification.

    Use the ``n x n`` symmetric distance matrix `D` and target class
    labels `target` to perform a `k`-NN experiment (leave-one-out
    cross-validation or evaluation of test set; see parameter `test_ind`).
    Ties are broken by the nearest neighbor.

    Parameters
    ----------
    D : ndarray
        The ``n x n`` symmetric distance (similarity) matrix.

    target : ndarray (of dtype=int)
        The ``n x 1`` target class labels (ground truth) or
        ``n x c`` in case of ``c`` binarized multilabels

    k : int or array_like (of dtype=int), optional (default: 5)
        Neighborhood size for `k`-NN classification.
        For each value in `k`, one `k`-NN experiment is performed.

        HINT: Providing more than one value for `k` is a cheap means to perform
        multiple `k`-NN experiments at once. Try e.g. ``k=[1, 5, 20]``.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Perform a LOO-CV experiment
        - ndarray : Hold out points indexed in this array as test set. Fit
          model to remaining data. Evaluate model on test set.

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    return_cmat : bool, optional, default: True
        If False, only return the predictions `y_pred`.
        Otherwise also return the confusion matrices.

    Returns
    -------
    y_pred : ndarray (shape=(n_k, n, c), dtype=int)
        Predicted class labels (`n_k`... number of items in parameter `k`)
        
        HINT: Referring to the above example... 
        ... ``y_pred[0]`` gives the predictions of the ``k=1`` experiment.

    cmat : ndarray (shape=(n_k x c x n_t x n_t), dtype=int) 
        Confusion matrix (``n_t`` number of unique items in parameter target)

        HINT: ... ``cmat[2, 0, :, :]`` gives the confusion matrix of
        the first class in the ``k=20`` experiment in the following order:
            TN    FP
            FN    TP
    """

    # Check input sanity
    log = Logging.ConsoleLogging()
    if sample_idx is None:
        IO.check_distance_matrix_shape(D)
    else:
        IO.check_sample_shape_fits(D, sample_idx)
    #IO._check_distance_matrix_shape_fits_labels(D, target)
    IO.check_valid_metric_parameter(metric)
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
    if metric == 'similarity':
        d_self = -np.inf
        sort_order = -1

    # Copy, because data is changed
    if not issparse(D):
        D = D.copy()
    target = target.astype(int)
    if target.ndim == 1:
        target = target[:, np.newaxis]
    if verbose:
        log.message("Start k-NN experiment.")
    # Handle LOO-CV vs. test set mode
    if test_ind is None:
        n = D.shape[0]
        test_set_ind = range(n)    # dummy     IO.check_valid_metric_parameter(metric)
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

    cl = np.sort(np.unique(target))
    cmat = np.zeros((k_length, target.shape[1], len(cl), len(cl)), dtype=int)
    y_pred = np.zeros((k_length, *target.shape), dtype=int)

    classes = target.copy()
    for idx, cur_class in enumerate(np.array(cl).ravel()):
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
        if verbose and ((i+1)%1000==0 or i+1==n):
            log.message("Prediction: {} of {}.".format(i+1, n), flush=True)

        if issparse(D):
            row = D.getrow(i)
            #row = D.data
            ind = row.nonzero()[1]
            row = row.toarray().ravel()
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
            if issparse(D):
                rp = ind
            else:
                rp = np.arange(len(sample_idx))
        rp = np.random.permutation(rp)
        d2 = row[rp]
        d2idx = np.argsort(d2, axis=0)[::sort_order]
        d2idx = d2idx[~np.isnan(d2[d2idx])] # filter NaN values
        idx = rp[d2idx]

        # More than one k is useful for cheap multiple k-NN experiments at once
        for j in range(k_length):
            # Make sure no inf/-inf/nan values are used for classification
            finite_val = np.isfinite(row[idx[0:k[j]]])
            # However, if no values are finite, classify randomly
            if finite_val.sum() == 0:
                idx = np.random.permutation(idx)
                finite_val = np.ones_like(finite_val)
                log.warning("Query was classified randomly, because all "
                            "distances were non-finite numbers.")
            for l in range(target.shape[1]):
                l_classes = classes[:, l]
                if sample_idx is None:
                    nn_class = l_classes[idx[0:k[j]]][finite_val]
                else:
                    l_sample_classes = sample_classes[:, l]
                    nn_class = l_sample_classes[idx[0:k[j]]][finite_val]
                cs = np.bincount(nn_class.astype(int))
                max_cs = np.where(cs == np.max(cs))[0]
                seed_class = classes[i, l]
                # "tie": use nearest neighbor
                if len(max_cs) > 1:
                    y_pred[j, i, l] = nn_class[0]
                    cmat[j, l, seed_class, nn_class[0]] += 1
                # majority vote
                else:
                    y_pred[j, i, l] = cl[max_cs[0]]
                    cmat[j, l, seed_class, cl[max_cs[0]]] += 1

    if verbose:
        log.message("Finished k-NN experiment.")

    if return_cmat:
        return y_pred, cmat
    else:
        return y_pred

##############################################################################
#
#  R - PRECISION
#
def _load_shared_csr(shared_data_, shared_indices_, 
                     shared_indptr_, shape_, n_rnd_pred_,
                     shared_rel_items_, shared_y_):
    global S
    S = csr_matrix((np.frombuffer(shared_data_),
                    np.frombuffer(shared_indices_), 
                    np.frombuffer(shared_indptr_)), 
                   shape=shape_, copy=False)
    global n_rnd_pred
    n_rnd_pred = n_rnd_pred_
    global relevant_items
    relevant_items = np.frombuffer(shared_rel_items_, dtype=int)
    global y
    y = np.frombuffer(shared_y_, dtype=int)


def _r_prec_worker(i, y_pred, incorrect, **kwargs):
    # = args
    true_class = y[i]
    if y_pred:
        nn_labels = np.zeros(y_pred, dtype=int) + incorrect
    if relevant_items[true_class] == 0:
        if y_pred:
            return 0., nn_labels
        else:
            return 0. # there can't be correct predictions...

    # Get all nonzero similarities
    row = S.getrow(i).copy() # .copy() should be redundant
    nnz = row.nnz
    # Randomize to avoid problems arising from equal similarites
    rp = np.random.permutation(nnz)
    d2 = row.data[rp]
    # Partition for each k value
    kth = nnz - relevant_items[true_class] - 1
    # sort the two highest similarities to end
    kth = np.append(kth, [nnz-2, nnz-1])
    # Clip negative indices (nnz < k)
    np.clip(kth, a_min=0, a_max=nnz-1, out=kth)
    # Remove duplicate k values and sort
    kth = np.unique(kth)
    # Get the relevant indices
    d2idx = np.argpartition(d2, kth=kth)
    # Filter indices pointing to NaN values and revert order
    d2idx = d2idx[~np.isnan(d2[d2idx])][::-1]
    # Indices of cells with highest similarities
    idx = row.nonzero()[1][rp[d2idx]]
    # Remove self similarity (i.e. highest sim)
    idx = idx[1:relevant_items[true_class]+1]
    # Check whether the values are finite
    finite_val = np.isfinite(row[0, idx].toarray().ravel())

    # However, if no values are finite, classify randomly
    if finite_val.sum() == 0:
        idx = np.random.permutation(idx)
        finite_val = np.ones_like(finite_val)
        with n_rnd_pred.get_lock():
            n_rnd_pred.value += 1
    
    y_predicted = y[idx]
    correct_pred = (y_predicted == true_class).sum()
    if correct_pred == 0:
        if y_pred:
            return 0., nn_labels
        else:
            return 0.
    else:
        if y_pred:
            cur_nn_labels = y_predicted[:y_pred]
            nn_labels[:cur_nn_labels.size] = cur_nn_labels
            for i, label in enumerate(cur_nn_labels):
                nn_labels[i] = label
            return correct_pred / relevant_items[true_class], nn_labels
        else:
            return correct_pred / relevant_items[true_class]


def r_precision(S:np.ndarray, y:np.ndarray, metric:str='distance',
                average:str='weighted', return_y_pred:int=0,
                verbose:int=0, n_jobs:int=1) -> float:
    ''' Calculate R-Precision (recall at R-th position).

    Parameters
    ----------
    S : ndarray or CSR matrix
        Distance (similarity) matrix

    y : ndarray
        Target (ground truth) labels

    metric : 'distance' or 'similarity', optional, default: 'similarity'
        Define, whether `S` is a distance or similarity matrix.

    average : 'weighted', 'macro' or None, optional, default: 'weighted'
        Ignored. Weighted and macro precisions are returned.

    return_y_pred : int, optional, default: 0
        If > 0, return the labels of the `return_y_pred` nearest neighbors

    verbose : int, optional, default: 0
        Increasing level of output.

    n_jobs : int, optional, default: 1
        Number of parallel processes to use.

    Returns
    -------
    r_precision : dictionary with following keys:
        macro : float
            Macro R-Precision.

        weighted : float
            Weighted R-Precision.

        per_item : ndarray
            R-Precision at the object.

        relevant_items : ndarray
            Relevant items per class.

        y_true : ndarray
            Target labels (req. for weighting).

        y_pred : ndarray
            Labels of some k-nearest neighbors
    '''
    IO.check_distance_matrix_shape(S)
    IO.check_distance_matrix_shape_fits_labels(S, y)
    IO.check_valid_metric_parameter(metric)
    log = Logging.ConsoleLogging()
    n, _ = S.shape
    S_is_sparse = issparse(S)
    if metric != 'similarity' or not S_is_sparse:
        raise NotImplementedError("Only sparse similarity matrices so far.")

    # Map labels to 0..n(labels)-1
    le = LabelEncoder()
    # Add int.min for misclassifications
    incorr_orig = np.array([np.nan]).astype(int)
    le.fit(np.append(y, incorr_orig))
    y = le.transform(y)
    incorrect = le.transform(incorr_orig)
    # Number of relevant items, i.e. number of each label
    relevant_items = np.bincount(y) - 1 # one less for self class
    # R-Precision for each item
    r_prec = np.zeros(y.shape, dtype=np.float)
    
    
    # Classify each point in test set
    if verbose:
        log.message("Creating shared memory data.")
    shared_data = mp.RawArray(ctypes.c_double, S.data.size)
    shared_data_np = np.frombuffer(shared_data)
    shared_data_np[:] = S.data[:]
    shared_indices = mp.RawArray(ctypes.c_double, S.indices.size)
    shared_indices_np = np.frombuffer(shared_indices)
    shared_indices_np[:] = S.indices[:]
    shared_indptr = mp.RawArray(ctypes.c_double, S.indptr.size)
    shared_indptr_np = np.frombuffer(shared_indptr)
    shared_indptr_np[:] = S.indptr[:]
    n_random_pred = mp.Value(ctypes.c_int)
    n_random_pred.value = 0
    shared_rel_items = mp.RawArray(ctypes.c_int64, relevant_items.size)
    shared_rel_items_np = np.frombuffer(shared_rel_items, dtype=int)
    shared_rel_items_np[:] = relevant_items[:]
    shared_y = mp.RawArray(ctypes.c_int64, y.size)
    shared_y_np = np.frombuffer(shared_y, dtype=int)
    shared_y_np[:] = y[:]

    if verbose and log:
        log.message("Spawning processes for prediction.")
    y_pred = np.zeros((n, return_y_pred), dtype=float)
    kwargs = {#'relevant_items' : relevant_items,
              #'y' : y,
              'y_pred' : return_y_pred,
              'incorrect' : incorrect}
    with mp.Pool(processes=n_jobs, initializer=_load_shared_csr, 
              initargs=(shared_data, shared_indices, 
                        shared_indptr, S.shape, n_random_pred,
                        shared_rel_items, shared_y)) as pool:
        for i, r in enumerate(
            pool.imap(
                func=partial(_r_prec_worker, **kwargs),
                iterable=range(n), 
                chunksize=int(1e2))):
            if verbose and ((i+1)%int(1e7 / 10**verbose) == 0 or i == n-1):
                log.message("Classification: {} of {} on {}.".format(
                            i+1, n, mp.current_process().name), flush=True)
            try:
                r_prec[i] = r[0]
                y_pred[i, :] = r[1]
            except:
                r_prec[i] = r
    pool.join()

    if verbose and log:
        log.message("Retrieving nearest neighbors.")
    y_pred = le.inverse_transform(y_pred.astype(int))
    if verbose and log:
        log.message("Finishing.")
    if n_random_pred.value:
        log.warning(("{} queries were classified randomly, because all "
            "distances were non-finite numbers or there were no other "
            "objects in the same class.").format(n_random_pred.value))
    return_dict = {'macro' : r_prec.mean(),
                   'weighted' : np.average(r_prec, weights=relevant_items[y]),
                   'per_item' : r_prec,
                   'relevant_items' : relevant_items,
                   'y_true' : y,
                   'y_pred' : y_pred}
    return return_dict

def f1_score(cmat):
    ''' Calculate F measure from confusion matrix.

    Parameters
    ----------
    cmat : ndarray
        Confusion matrix.
        
        Assuming confusion matrix in following format:
            TN    FP
            FN    TP
        E.g. as obtained from predict(...)[0, 0, :, :]

    Returns
    -------
    f1 : float
        F measure of the given confusion matrix.
    '''
    #TN = cmat[0, 0] not required for F1 score
    FP = cmat[0, 1]
    FN = cmat[1, 0]
    TP = cmat[1, 1]
    if TP == 0: # pathological case
        return 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return 2 * precision * recall / (precision + recall)

def f1_macro(cmat):
    ''' Calculate macro averaged F measure from confusion matrices.

    Biased towards LESS abundant classes in case of class imbalance.

    Parameters
    ----------
    cmat : ndarray
        3D Confusion matrix. Format as obtained from predict(...)[0, :, :, :]

    Returns
    -------
    f1_macro : float
        Macro F measure of the given confusion matrices.
    '''
    return np.array([f1_score(x) for x in cmat]).mean()

def f1_weighted(cmat):
    ''' Calculate weighted F measure from confusion matrices.

    Parameters
    ----------
    cmat : ndarray
        3D Confusion matrix. Format as obtained from predict(...)[0, :, :, :]

    Returns
    -------
    f1_weighted : float
        Weighted F measure of the given confusion matrices.
    '''
    scores = np.array([f1_score(x) for x in cmat])
    weights = np.array([x[1, :].sum() for x in cmat])
    return np.average(scores, weights=weights)

def f1_micro(cmat):
    ''' Calculate micro averaged F measure from confusion matrices.
    
    Biased towards MORE abundant classes in case of class imbalance.

    Parameters
    ----------
    cmat : ndarray
        3D Confusion matrix. Format as obtained from predict(...)[0, :, :, :]

    Returns
    -------
    f1_micro : float
        Micro F measure of the given confusion matrices.
    '''
    return f1_score(cmat.sum(axis=0))
