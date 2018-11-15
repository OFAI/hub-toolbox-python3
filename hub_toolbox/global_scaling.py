#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2018, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
from functools import partial
from itertools import filterfalse
import ctypes
import numpy as np
from scipy.special import gammainc  # @UnresolvedImport
from scipy.stats import norm
from scipy.sparse import lil_matrix, csr_matrix, issparse
from multiprocessing import Pool, cpu_count, current_process
from multiprocessing.sharedctypes import Array
from hub_toolbox import io
from hub_toolbox.htlogging import ConsoleLogging

__all__ = ['mutual_proximity_empiric', 'mutual_proximity_gammai', 
           'mutual_proximity_gaussi', '_mutual_proximity_gumbel_sparse']

def mutual_proximity_empiric(D:np.ndarray, metric:str='distance',
                             test_ind:np.ndarray=None, verbose:int=0,
                             sample_ind:np.ndarray=None, n_jobs=None,
                             min_nnz:int=0):
    """Transform a distance matrix with Mutual Proximity (empiric distribution).

    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix using
    the empiric data distribution (EXACT, rather SLOW). The resulting
    secondary distance/similarity matrix should show lower hubness.

    Parameters
    ----------
    D : ndarray
        Distance or similarity matrix.

        - Shape ``n x n`` matrix for scaling the complete matrix.
        - Shape ``n x s`` matrix (where ``n`` and ``s`` are the dataset and
        sample size, respectively) for scaling only the distances to
        ``s`` sample instances, whose indices must be specified with
        parameter `sample_ind`.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    sample_ind : ndarray
        The index array that determines, which data points the columns in
        `D` correspond to (indices of training data).

    Returns
    -------
    D_mp : ndarray
        Secondary distance MP empiric matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    if sample_ind is None:
        return _mutual_proximity_empiric_full(D=D, metric=metric,
                                              test_set_ind=test_ind,
                                              min_nnz=min_nnz,
                                              verbose=verbose,
                                              n_jobs=n_jobs)
    else:
        return _mutual_proximity_empiric_sample(D=D, idx=sample_ind,
                                                metric=metric,
                                                test_set_ind=test_ind,
                                                verbose=verbose,
                                                n_jobs=n_jobs)

def _mutual_proximity_empiric_sample(D:np.ndarray, idx:np.ndarray,
    metric:str='distance', test_set_ind:np.ndarray=None,
    verbose:int=0, n_jobs=None):
    """Transform a distance matrix with Mutual Proximity (empiric distribution).
    
    NOTE: this docstring does not yet fully reflect the properties of this
    proof-of-concept function!
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix using
    the empiric data distribution (EXACT, rather SLOW). The resulting
    secondary distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray
        The ``n x s`` distance or similarity matrix, where ``n`` and ``s``
        are the dataset and sample size, respectively.

    idx : ndarray
        The index array that determines, to which data points the columns in
        `D` correspond.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    Returns
    -------
    D_mp : ndarray
        Secondary distance MP empiric matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    # Initialization and checking input
    log = ConsoleLogging()
    io.check_sample_shape_fits(D, idx)
    io.check_valid_metric_parameter(metric)
    n = D.shape[0]
    s = D.shape[1]
    if metric == 'similarity':
        self_value = 1
        exclude_value = np.inf
    else: # metric == 'distance':
        self_value = 0
        exclude_value = -np.inf
        if issparse(D):
            raise ValueError("MP sparse only supports similarity matrices.")
    if test_set_ind is None:
        n_ind = range(n)
    #elif not np.all(~test_set_ind):
    else:
        n_ind = test_set_ind
        #raise NotImplementedError("MP empiric does not yet support train/"
        #                          "test splits.")
        #train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    # Start MP
    D = D.copy()
    
    if issparse(D):
        raise NotImplementedError
        #return _mutual_proximity_empiric_sparse(D, test_set_ind, verbose, log)
        
    # ensure correct self distances (NOT done for sparse matrices!)
    for j, sample in enumerate(idx):
        D[sample, j] = exclude_value

    D_mp = np.zeros_like(D) * np.nan
     
    # Calculate MP empiric
    for i in n_ind: #range(n):
        if verbose and ((i+1)%1000 == 0 or i == n-2):
            log.message("MP_empiric: {} of {}.".format(i+1, n-1), flush=True)
        dI = D[i, :][np.newaxis, :] # broadcasted afterwards
        dJ = D[idx, :] # fancy indexing, thus copy
        d = dI.T # D[i, :][:, np.newaxis] # both versions are equal
        # div by n
        n_pts = s 
        # div by n-1, n-2 #n_pts = (np.isfinite(dI) & np.isfinite(dJ)).sum(1)
        if metric == 'similarity':
            D_mp[i, :] = np.sum((dI <= d) & (dJ <= d), 1) / n_pts
        else: # metric == 'distance':
            D_mp[i, :] = 1 - (np.sum((dI > d) & (dJ > d), 1) / n_pts)

    # Ensure correct self distances
    for j, sample in enumerate(idx):
        D_mp[sample, j] = self_value
    
    if test_set_ind is None:
        return D_mp
    else:
        return D_mp[test_set_ind]

def _mutual_proximity_empiric_full(D:np.ndarray, metric:str='distance', 
                                  test_set_ind:np.ndarray=None, min_nnz:int=0,
                                  verbose:int=0, n_jobs=None):
    """Transform a distance matrix with Mutual Proximity (empiric distribution).
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix using 
    the empiric data distribution (EXACT, rather SLOW). The resulting 
    secondary distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray or csr_matrix
        - ndarray: The ``n x n`` symmetric distance or similarity matrix.
        - csr_matrix: The ``n x n`` symmetric similarity matrix.
          
        NOTE: In case of sparse ``D`, zeros are interpreted as missing values 
        and ignored during calculations. Thus, results may differ 
        from using a dense version.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.
        
        NOTE: In case of sparse `D`, only 'similarity' is supported.
        
    test_set_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 

    min_nnz : int, optional, default: 0
        Calculate MP between two objects `i` and `j`, iff at least ``min_nnz``
        values are present in both row ``i`` and ``j``.
        Otherwise, return the original distance/similarity.
        
        NOTE: Currently only implemented for MP empiric w/ sparse sim matrices

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
        
    Returns
    -------
    D_mp : ndarray
        Secondary distance MP empiric matrix.
    
    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
           Local and global scaling reduce hubs in space. The Journal of Machine 
           Learning Research, 13(1), 2871–2902.
    """
    # Initialization
    n = D.shape[0]
    log = ConsoleLogging()
    
    # Check input
    io.check_distance_matrix_shape(D)
    io.check_valid_metric_parameter(metric)
    if metric == 'similarity':
        self_value = 1
        exclude_value = np.inf
    else: # metric == 'distance':
        self_value = 0
        exclude_value = -np.inf
        if issparse(D):
            raise ValueError("MP sparse only supports similarity matrices.")
    if test_set_ind is None:
        pass # TODO implement
        #train_set_ind = slice(0, n)
    elif not np.all(~test_set_ind):
        raise NotImplementedError("MP empiric does not yet support train/"
                                  "test splits.")
        #train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    if issparse(D):
        return _mutual_proximity_empiric_sparse(D, test_set_ind, min_nnz, verbose, log, n_jobs)
    # Start MP
    D = D.copy()
    
        
    # ensure correct self distances (NOT done for sparse matrices!)
    np.fill_diagonal(D, exclude_value)
    
    D_mp = np.zeros_like(D)
     
    # Calculate MP empiric
    for i in range(n-1):
        if verbose and ((i+1)%1000 == 0 or i == n-2):
            log.message("MP_empiric: {} of {}.".format(i+1, n-1), flush=True)
        # Calculate only triu part of matrix
        j_idx = i + 1
         
        dI = D[i, :][np.newaxis, :]
        dJ = D[j_idx:n, :]
        d = D[j_idx:n, i][:, np.newaxis]
         
        if metric == 'similarity':
            D_mp[i, j_idx:] = np.sum((dI <= d) & (dJ <= d), 1) / n #(n - 2)
        else: # metric == 'distance':
            D_mp[i, j_idx:] = 1 - (np.sum((dI > d) & (dJ > d), 1) / n) #(n - 2))
         
    # Mirror, so that matrix is symmetric
    D_mp += D_mp.T
    np.fill_diagonal(D_mp, self_value)

    return D_mp

#==============================================================================
# #============================================================================
# #                             MP empiric sparse
# #============================================================================
#==============================================================================
def _mpes_init(S_, shared_data_):
    global S
    S = S_
    global S_mp_data
    S_mp_data = shared_data_

def _mpes_sec_dist(ind, args):
    """Compute MP between two objects i and j in CSR matrix."""
    i, j = ind
    verbose, log, n, min_nnz = args

    if verbose:
        n_rows = int(1e5 / 10**verbose)
    if verbose and log and i==j and ((i+1)%n_rows == 0 or i == n-1):
        log.message("MP_empiric: {} of {} on {}.".format(
            i+1, n, current_process().name), flush=True)
    # Original similarity between the two objects
    threshold = S[i, j]

    # Find the corresponding index in S.data (or S_mp.data) in O(log(n)) time.
    data_idx = S.indptr[i] + np.searchsorted(S.indices[S.indptr[i]:S.indptr[i+1]], j)
    
    # Similarities to i/j (as sparse matrices (rows))
    S_i = S.getrow(i)
    S_j = S.getrow(j)

    # If any row contains too few values, 
    # just return the original similarities (handled elsewhere).
    if S_i.nnz <= min_nnz or S_j.nnz <= min_nnz:
        with S_mp_data.get_lock():
            S_mp_data[data_idx] = np.nan
        return #i, j, np.nan
    # otherwise count those positions lte to `s`` in both rows
    else:
        S_i.data[S_i.data <= threshold] = 0
        S_j.data[S_j.data <= threshold] = 0
        s_mp = 1 - (S_i + S_j).nnz / n
        with S_mp_data.get_lock():
            S_mp_data[data_idx] = s_mp
        return #i, j, s_mp

def _mutual_proximity_empiric_sparse(S:csr_matrix,
                                     test_set_ind:np.ndarray=None, 
                                     min_nnz=0,
                                     verbose:int=0,
                                     log=None,
                                     n_jobs=None):
    """MP empiric for sparse similarity matrices.

    Please do not directly use this function, but invoke via 
    mutual_proximity_empiric()
    """
    if verbose and log:
        log.message("Starting MP empiric for sparse matrices.")
    self_value = 1. # similarity matrix
    n = S.shape[0]
    if not n_jobs:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = cpu_count()
    else:
        pass

    # This will become S_mp.data
    shared_data = Array(ctypes.c_double, S.data.size)
    shared_data_np = np.ctypeslib.as_array(shared_data.get_obj())

    if verbose and log:
        log.message("Spawning processes and starting MP computation.")
    with Pool(processes=n_jobs,
              initializer=_mpes_init,
              initargs=(S, shared_data)) as pool:  
        S_nonzero = filterfalse(lambda ij: ij[0] > ij[1], zip(*S.nonzero()))
        for _ in pool.imap(
            func=partial(_mpes_sec_dist, args=(verbose, log, n, min_nnz)),
            iterable=S_nonzero,
            chunksize=int(1e5)):
            pass # output stored by function in shared array
    pool.join()
    if verbose and log:
        log.message("Assemble upper-triangular MP matrix.")
    S_mp = csr_matrix((shared_data_np, S.indices, S.indptr),
                      shape=S.shape, copy=False).tolil()
    del shared_data, shared_data_np
    if verbose and log:
        log.message("Symmetrizing matrix.")
    S_mp += S_mp.T
    # Retain original distances for objects with too few neighbors.
    # That is, keep distances FROM these objects to others (rows), but
    # set distances of other objects TO them to NaN (columns).
    # Returned matrix is thus NOT SYMMETRIC.
    if verbose and log:
        log.message(("Retain original similarities for objects with too few "
                     "neighbors. If there are any, the output matrix will "
                     "not be symmetric anymore! (Rows corresponding to these "
                     "objects will be in original space; corresponding "
                     "columns will contain NaN)."))
    for row in np.argwhere(S.getnnz(axis=1) <= min_nnz):
        row = row[0] # use scalar for indexing instead of array
        S_mp[row, :] = S.getrow(row)
    if verbose and log:
        log.message("Setting self similarities.")
    for i in range(n):
        S_mp[i, i] = self_value #need to set self values
    if verbose and log:
        log.message("Converting to CSR matrix and returning.")
    return S_mp.tocsr()


def mutual_proximity_gaussi_sample(D:np.ndarray, idx:np.ndarray, 
    metric:str='distance', test_set_ind:np.ndarray=None, verbose:int=0):
    """Transform a distance matrix with Mutual Proximity (empiric distribution).
    
    NOTE: this docstring does not yet fully reflect the properties of this 
    proof-of-concept function!
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix using 
    the empiric data distribution (EXACT, rather SLOW). The resulting 
    secondary distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray
        The ``n x s`` distance or similarity matrix, where ``n`` and ``s``
        are the dataset and sample size, respectively.

    idx : ndarray
        The index array that determines, to which data points the columns in
        `D` correspond.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

    test_set_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    Returns
    -------
    D_mp : ndarray
        Secondary distance MP empiric matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    # Initialization and checking input
    log = ConsoleLogging()
    io.check_sample_shape_fits(D, idx)
    io.check_valid_metric_parameter(metric)
    n = D.shape[0]
    s = D.shape[1]
    j = np.ones(n, int)
    j *= (n+1) # illegal indices will throw index out of bounds error
    j[idx] = np.arange(s)
    if metric == 'similarity':
        self_value = 1
    else: # metric == 'distance':
        self_value = 0
    exclude_value = np.nan
    if test_set_ind is None:
        n_ind = range(n)
    else:
        n_ind = test_set_ind

    # Start MP
    D = D.copy()

    if issparse(D):
        raise NotImplementedError

    # ensure correct self distances (NOT done for sparse matrices!)
    for j, sample in enumerate(idx):
        D[sample, j] = exclude_value

    # Calculate mean and std per row, w/o self values (nan)
    mu = np.nanmean(D, 1)
    sd = np.nanstd(D, 1, ddof=0)
    # Avoid downstream div/0 errors
    sd[sd == 0] = 1e-7

    # set self dist/sim back to self_value to avoid scipy warnings
    for j, i in enumerate(idx):
        D[i, j] = self_value

    # MP Gaussi
    D_mp = np.zeros_like(D)
    for sample, i in enumerate(n_ind):
        if verbose and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        j = slice(0, s)
        
        if metric == 'similarity':
            p1 = norm.cdf(D[i, j], mu[i], sd[i])
            p2 = norm.cdf(D[i, j], mu[idx], sd[idx])
            D_mp[i, j] = (p1 * p2).ravel()
        else:
            # Survival function: sf(.) := 1 - cdf(.)
            p1 = norm.sf(D[i, j], mu[i], sd[i])
            p2 = norm.sf(D[i, j], mu[idx], sd[idx])
            D_mp[i, j] = (1 - p1 * p2).ravel()

    # Ensure correct self distances
    for j, sample in enumerate(idx):
        D_mp[sample, j] = self_value

    if test_set_ind is None:
        return D_mp
    else:
        return D_mp[test_set_ind]

def mutual_proximity_gaussi(D:np.ndarray, metric:str='distance',
                            sample_size:int=0, min_nnz:int=30,
                            test_set_ind:np.ndarray=None,
                            verbose:int=0, idx:np.ndarray=None,):
    """Transform distances with Mutual Proximity (indep. normal distributions).
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix. Gaussi 
    variant assumes independent normal distributions (FAST).
    The resulting second. distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray or csr_matrix
        - ndarray: The ``n x n`` symmetric distance or similarity matrix.
        - csr_matrix: The ``n x n`` symmetric similarity matrix.
        
        NOTE: In case of sparse `D`, zeros are interpreted as missing values 
        and ignored during calculations. Thus, results may differ 
        from using a dense version.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.
        
        NOTE: In case of sparse `D`, only 'similarity' is supported.
        
    sample_size : int, optional (default: 0)
        Define sample size from which Gauss parameters are estimated.
        Use all data when set to ``0``.
        Ignored in case of SampleMP (i.e. if provided `idx`).

    min_nnz : int, optional, default: 30
        Calculate MP between two objects `i` and `j`, iff at least ``min_nnz``
        values are present in both row ``i`` and ``j``.
        Otherwise, return the original similarity.
        Ignored, if `metric` is 'distance'.

    test_set_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

        Ignored in case of SampleMP (i.e. if provided `idx`).

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    idx : ndarray, optional (default: None)
        The index array that determines to which data points the columns in
        `D` correspond. Only required for SampleMP.

    Returns
    -------
    D_mp : ndarray
        Secondary distance MP gaussi matrix.
    
    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
           Local and global scaling reduce hubs in space. The Journal of Machine 
           Learning Research, 13(1), 2871–2902.
    """    
    # Initialization   
    log = ConsoleLogging()

    # Checking input
    if idx is None:
        io.check_distance_matrix_shape(D)
    else:
        io.check_sample_shape_fits(D, idx)
    io.check_valid_metric_parameter(metric)
    n = D.shape[0]
    s = D.shape[1]

    if metric == 'similarity':
        self_value = 1
    else: # metric == 'distance':
        self_value = 0  
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    # Start MP Gaussi    
    if verbose:
        log.message('Mutual Proximity Gaussi rescaling started.', flush=True)
    D = D.copy()

    if issparse(D):
        return _mutual_proximity_gaussi_sparse(D, sample_size, min_nnz,
                                               test_set_ind, verbose, log)

    # ignore self dist/sim for parameter estimation
    if idx is None:
        np.fill_diagonal(D, np.nan)
    else:
        for j, i in enumerate(idx):
            D[i, j] = np.nan

    # Calculate mean and std
    if idx is None:
        if sample_size == 0:
            mu = np.nanmean(D[train_set_ind], 0)
            sd = np.nanstd(D[train_set_ind], 0, ddof=0)
        else:
            samples = np.random.shuffle(train_set_ind)[0:sample_size]
            mu = np.nanmean(D[samples], 0)
            sd = np.nanstd(D[samples], 0, ddof=0)
    else:
        mu = np.nanmean(D, 1)
        sd = np.nanstd(D, 1, ddof=0)
    # Avoid downstream div/0 errors
    sd[sd == 0] = 1e-7
    # set self dist/sim back to self_value to avoid scipy warnings
    if idx is None:
        np.fill_diagonal(D, self_value)
    else:
        for j, i in enumerate(idx):
            D[i, j] = self_value

    # MP Gaussi
    D_mp = np.zeros_like(D)
    for i in range(n):
        if verbose and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        if idx is None:
            j = slice(i+1, n)
            j_mom = j
        else:
            j = slice(0, s)
            j_mom = idx[j]
        
        if metric == 'similarity':
            p1 = norm.cdf(D[i, j], mu[i], sd[i])
            p2 = norm.cdf(D[i, j], mu[j_mom], sd[j_mom])
            D_mp[i, j] = (p1 * p2).ravel()
        else:
            # sf(.) := 1 - cdf(.)
            p1 = norm.sf(D[i, j], mu[i], sd[i])
            p2 = norm.sf(D[i, j], mu[j_mom], sd[j_mom])
            D_mp[i, j] = (1 - p1 * p2).ravel()

    if idx is None:
        D_mp += D_mp.T        
        np.fill_diagonal(D_mp, self_value)
    else:
        # Ensure correct self distances
        for j, sample in enumerate(idx):
            D_mp[sample, j] = self_value
    return D_mp

def _mutual_proximity_gaussi_sparse(S:np.ndarray, sample_size:int=0,
                                    min_nnz:int=30,
                                    test_set_ind:np.ndarray=None, 
                                    verbose:int=0, log=None):
    """MP gaussi for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gaussi()
    """
    n = S.shape[0]
    self_value = 1 # similarity matrix
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
    #===========================================================================
    # from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  # @UnresolvedImport
    # mu, var = csr_mean_variance_axis0(S[train_set_ind])
    # sd = np.sqrt(var)
    # del var
    #===========================================================================
    
    # mean, variance WITHOUT zero values (missing values), ddof=0
    if S.diagonal().max() != 1. or S.diagonal().min() != 1.:
        raise ValueError("Self similarities must be 1.")
    S_param = S[train_set_ind]
    # the -1 accounts for self similarities that must be excluded from the calc
    mu = np.array((S_param.sum(0) - 1.) / (S_param.getnnz(0) - 1)).ravel()
    X = S_param
    X.data **= 2
    E1 = np.array((X.sum(0) - 1.) / (X.getnnz(0) - 1)).ravel()
    del X, S_param
    E2 = mu**2
    va = E1 - E2
    del E1, E2
    sd = np.sqrt(va)
    del va
    
    S_mp = lil_matrix(S.shape)
    #nnz = S.getnnz(axis=1) # nnz per row

    for i in range(n):
        if verbose and log and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)
        S_ij = S[i, j_idx].toarray().ravel() #Extract dense rows temporarily
        tmp = np.empty(n-i)
        tmp[0] = self_value / 2.
        #=======================================================================
        # if nnz[i] <= min_nnz:
        #     tmp[1:] = S_ij
        # else: # Only rescale, if there are sufficient neighbors
        #=======================================================================
        S_ji = S[j_idx, i].toarray().ravel() #for vectorization below.

        p1 = norm.cdf(S_ij, mu[i], sd[i]) # mu, sd broadcasted
        p1[S_ij == 0] = 0
        del S_ij
        p2 = norm.cdf(S_ji, mu[j_idx], sd[j_idx])
        p2[S_ji == 0] = 0
        del S_ji
        tmp[1:] = (p1 * p2).ravel()
        S_mp[i, i:] = tmp         
        del tmp, j_idx

    S_mp += S_mp.T
    return S_mp.tocsr()

def mutual_proximity_gammai(D:np.ndarray, metric:str='distance',
                            min_nnz:int=30, test_set_ind:np.ndarray=None,
                            verbose:int=0):
    """Transform a distance matrix with Mutual Proximity (indep. Gamma distr.).
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix. Gammai 
    variant assumes independent Gamma distributed distances (FAST).
    The resulting second. distance/similarity matrix should show lower hubness.

    Parameters
    ----------
    D : ndarray or csr_matrix
        - ndarray: The ``n x n`` symmetric distance or similarity matrix.
        - csr_matrix: The ``n x n`` symmetric similarity matrix.
        
        NOTE: In case of sparse `D`, zeros are interpreted as missing values 
        and ignored during calculations. Thus, results may differ 
        from using a dense version.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

        NOTE: In case of sparse `D`, only 'similarity' is supported.

    min_nnz : int, optional, default: 30
        Calculate MP between two objects `i` and `j`, iff at least ``min_nnz``
        values are present in both row ``i`` and ``j``.
        Otherwise, return the original similarity.
        Ignored, if `metric` is 'distance'.

    test_set_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
 
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    Returns
    -------
    D_mp : ndarray
        Secondary distance MP gammai matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
           Local and global scaling reduce hubs in space. 
           The Journal of Machine Learning Research, 13(1), 2871–2902.
    """
    # Initialization
    n = D.shape[0]
    log = ConsoleLogging()

    # Checking input
    io.check_distance_matrix_shape(D)
    io.check_valid_metric_parameter(metric)
    if metric == 'similarity':
        self_value = 1
    else: # metric == 'distance':
        self_value = 0
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    # Start MP
    if verbose:
        log.message('Mutual proximity Gammai rescaling started.', flush=True)
    D = D.copy()
    
    if issparse(D):
        return _mutual_proximity_gammai_sparse(D, min_nnz, test_set_ind,
                                               verbose, log)

    np.fill_diagonal(D, np.nan)

    mu = np.nanmean(D[train_set_ind], 0)
    va = np.nanvar(D[train_set_ind], 0, ddof=1)
    # Avoid downstream div/0 errors
    va[va == 0] = 1e-7
    A = (mu**2) / va
    B = va / mu

    D_mp = np.zeros_like(D)

    # MP gammai
    for i in range(n):
        if verbose and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gammai: {} of {}".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)
        
        if metric == 'similarity':
            p1 = _local_gamcdf(D[i, j_idx], A[i], B[i])
            p2 = _local_gamcdf(D[j_idx, i], A[j_idx], B[j_idx])
            D_mp[i, j_idx] = (p1 * p2).ravel()
        else: # distance
            p1 = 1 - _local_gamcdf(D[i, j_idx], A[i], B[i])
            p2 = 1 - _local_gamcdf(D[j_idx, i], A[j_idx], B[j_idx])
            D_mp[i, j_idx] = (1 - p1 * p2).ravel()

    # Mirroring the matrix
    D_mp += D_mp.T
    # set correct self dist/sim
    np.fill_diagonal(D_mp, self_value)

    return D_mp

def _mutual_proximity_gammai_sparse(S:np.ndarray, min_nnz:int=30,
                                    test_set_ind:np.ndarray=None, 
                                    verbose:int=0, log=None):
    """MP gammai for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gammai()
    """
    n = S.shape[0]
    self_value = 1.
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    # mean, variance WITH zero values
    #=======================================================================
    # from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  
    # mu, va = csr_mean_variance_axis0(self.S[train_set_mask])
    #=======================================================================

    # mean, variance WITHOUT zero values (missing values), ddof=1
    if S.diagonal().max() != 1. or S.diagonal().min() != 1.:
        raise ValueError("Self similarities must be 1.")
    S_param = S[train_set_ind]
    # the -1 accounts for self similarities that must be excluded from the calc
    mu = np.array((S_param.sum(0) - 1) / (S_param.getnnz(0) - 1)).ravel()
    E2 = mu**2
    X = S_param.copy()
    X.data **= 2
    n_x = (X.getnnz(0) - 1)
    E1 = np.array((X.sum(0) - 1) / (n_x)).ravel()
    del X
    # for an unbiased sample variance
    va = n_x / (n_x - 1) * (E1 - E2)
    del E1

    A = E2 / va
    B = va / mu
    del mu, va, E2
    A[A < 0] = np.nan
    B[B <= 0] = np.nan

    S_mp = lil_matrix(S.shape, dtype=np.float32)
    nnz = S.getnnz(axis=1) # nnz per row

    for i in range(n):
        if verbose and log and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gammai: {} of {}".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)

        Dij = S[i, j_idx].toarray().ravel() #Extract dense rows temporarily
        tmp = np.empty(n-i)
        tmp[0] = self_value / 2. 
        if nnz[i] <= min_nnz:
            tmp[1:] = np.nan
        else:
            p1 = _local_gamcdf(Dij, A[i], B[i])
            del Dij
            Dji = S[j_idx, i].toarray().ravel() #for vectorization below.
            p2 = _local_gamcdf(Dji, A[j_idx], B[j_idx])
            del Dji
            tmp[1:] = (p1 * p2).ravel()
            S_mp[i, i:] = tmp     
            del tmp, j_idx
    S_mp += S_mp.T

    # Retain original distances for objects with too few neighbors.
    # That is, keep distances FROM these objects to others (rows), but
    # set distances of other objects TO them to NaN (columns).
    # Returned matrix is thus NOT SYMMETRIC.
    for row in np.argwhere(nnz <= min_nnz):
        row = row[0] # use scalar for indexing instead of array
        S_mp[row, :] = S.getrow(row)
    return S_mp.tocsr()

def _local_gamcdf(x, a, b, mv=np.nan):
    """Gamma CDF"""
    try:
        a[a < 0] = np.nan
    except TypeError:
        if a < 0:
            a = np.nan
    try:
        b[ b<= 0] = np.nan
    except TypeError:
        if b <= 0:
            b = np.nan
    x[x < 0] = 0

    # don't calculate gamcdf for missing values
    if mv == 0:
        nz = x > 0
        z = x[nz] / b[nz]
        p = np.zeros_like(x)
        p[nz] = gammainc(a[nz], z)
    else:
        z = x / b
        p = gammainc(a, z)
    return p

def _gumbelcdf(x, mu_hat, beta_hat):
    """Gumbel CDF"""
    return np.exp(-np.exp(-(x-mu_hat) / beta_hat))

def _mutual_proximity_gumbel_sparse(S:np.ndarray, min_nnz:int=30,
                                    test_set_ind:np.ndarray=None, 
                                    verbose:int=0, log=None):
    """MP Gumbel for sparse similarity matrices. 

    Please do not directly use this function, but invoke via 
    mutual_proximity_gumbel()
    """
    n = S.shape[0]
    self_value = 1.
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    # mean, variance WITHOUT zero values (missing values), ddof=1
    if S.diagonal().max() != 1. or S.diagonal().min() != 1.:
        raise ValueError("Self similarities must be 1.")
    S_param = S[train_set_ind]
    # the -1 accounts for self similarities that must be excluded from the calc
    mu = np.array((S_param.sum(0) - 1) / (S_param.getnnz(0) - 1)).ravel()
    E2 = mu**2
    X = S_param.copy()
    X.data **= 2
    n_x = (X.getnnz(0) - 1)
    E1 = np.array((X.sum(0) - 1) / (n_x)).ravel()
    del X
    # for an unbiased sample variance
    va = n_x / (n_x - 1) * (E1 - E2)
    del E1, E2
    sd = np.sqrt(va)
    del va

    # Euler-Mascheroni gamma=.57721566490153286 (https://oeis.org/A001620)
    EULER_MASCHERONI = np.euler_gamma
    beta_hat = sd * np.sqrt(6) / np.pi
    mu_hat = mu - EULER_MASCHERONI * beta_hat

    del mu, sd

    S_mp = lil_matrix(S.shape, dtype=np.float32)
    nnz = S.getnnz(axis=1) # nnz per row

    for i in range(n):
        if verbose and log and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gumbel: {} of {}".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)

        Dij = S[i, j_idx].toarray().ravel() #Extract dense rows temporarily        
        tmp = np.empty(n-i)
        tmp[0] = self_value / 2. 
        if nnz[i] <= min_nnz:
            tmp[1:] = np.nan
        else: # Rescale iff there are enough neighbors for current point
            p1 = _gumbelcdf(Dij, mu_hat[i], beta_hat[i])
            p1[Dij == 0] = 0.
            del Dij
            Dji = S[j_idx, i].toarray().ravel() #for vectorization below.
            p2 = _gumbelcdf(Dji, mu_hat[j_idx], beta_hat[j_idx])
            p2[Dji == 0] = 0.
            del Dji
            tmp[1:] = (p1 * p2).ravel()
            S_mp[i, i:] = tmp     
            del tmp, j_idx
    S_mp += S_mp.T

    # Retain original distances for objects with too few neighbors.
    # That is, keep distances FROM these objects to others (rows), but
    # set distances of other objects TO them to NaN (columns).
    # Returned matrix is thus NOT SYMMETRIC.
    for row in np.argwhere(nnz <= min_nnz):
        row = row[0] # use scalar for indexing instead of array
        S_mp[row, :] = S.getrow(row)
    return S_mp.tocsr()

if __name__ == '__main__':
    pass
