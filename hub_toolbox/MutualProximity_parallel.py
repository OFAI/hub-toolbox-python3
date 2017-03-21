#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2017, Dominik Schnitzer and Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import multiprocessing as mp
import numpy as np
from scipy.special import gammainc  # @UnresolvedImport
from scipy.stats import norm
from scipy.sparse import issparse, lil_matrix, csr_matrix
from hub_toolbox import IO, Logging
from hub_toolbox.Logging import ConsoleLogging

def mutual_proximity_empiric(D:np.ndarray, metric:str='distance', 
                             test_set_ind:np.ndarray=None, verbose:int=0,
                             n_jobs:int=-1):
    """Transform a distance matrix with Mutual Proximity (empiric distribution).
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix using 
    the empiric data distribution (EXACT, rather SLOW). The resulting 
    secondary distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray or csr_matrix
        - ndarray: The ``n x n`` symmetric distance or similarity matrix.
        - csr_matrix: The ``n x n`` symmetric similarity matrix.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.
        
        NOTE: In case of sparse `D`, only 'similarity' is supported.
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
        
    n_jobs : int, optional (default: -1)
        Number of parallel processes to be used.
        
        NOTE: set ``n_jobs=-1`` to use all CPUs
        
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
    log = Logging.ConsoleLogging()
    log.warning("MP parallel code is not up-to-date! "
                "Please use methods of the MutualProximity module for now.")
    IO.check_distance_matrix_shape(D)
    IO.check_valid_metric_parameter(metric)
    # DO NOT DELETE this comment, will be used upon parallel MP emp dist impl
    #===========================================================================
    # # Initialization
    # n = D.shape[0]
    # 
    # # Check input
    # if D.shape[0] != D.shape[1]:
    #     raise TypeError("Distance/similarity matrix is not quadratic.")        
    # if metric == 'similarity':
    #     self_value = 1
    # elif metric == 'distance':
    #     self_value = 0
    #     if issparse(D):
    #         raise ValueError("MP sparse only supports similarity matrices.")
    # else:
    #     raise ValueError("Parameter 'metric' must be 'distance' "
    #                      "or 'similarity'.")  
    # if test_set_ind is None:
    #     pass # TODO implement
    #     #train_set_ind = slice(0, n)
    # elif not np.all(~test_set_ind):
    #     raise NotImplementedError("MP empiric does not yet support train/"
    #                               "test splits.")
    #     #train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
    #===========================================================================
    if issparse(D):
        return _mutual_proximity_empiric_sparse(D, test_set_ind, verbose, log, n_jobs)
    else:
        log.warning("MP empiric does not support parallel execution for dense "
                    "matrices at the moment. Continuing with 1 process.")
        from hub_toolbox.MutualProximity import mutual_proximity_empiric
        return mutual_proximity_empiric(D, metric, test_set_ind, verbose)
    # DO NOT DELETE this comment
    #===========================================================================
    # # Start MP empiric
    # D = D.copy()
    # # ensure correct self distances (NOT done for sparse matrices!)
    # np.fill_diagonal(D, self_value)
    #===========================================================================
        
def _mutual_proximity_empiric_sparse(S:csr_matrix, 
                                     test_set_ind:np.ndarray=None, 
                                     verbose:int=0,
                                     log=None,
                                     n_jobs=-1):
    """MP empiric for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_empiric()
    """
    n = np.shape(S)[0]
    self_value = 1. # similarity matrix
    S_mp = lil_matrix(S.shape)
        
    # Parallelization
    if n_jobs == -1: # take all cpus
        NUMBER_OF_PROCESSES = mp.cpu_count()
    else:
        NUMBER_OF_PROCESSES = n_jobs
    tasks = []
    
    batches = _get_weighted_batches(n, NUMBER_OF_PROCESSES)
    
    for idx, batch in enumerate(batches):
        matrix = S
        tasks.append((_partial_mp_emp_sparse, (batch, matrix, idx, n, verbose)))
    
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    for task in tasks:
        task_queue.put(task)
        
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        processes.append(mp.Process(target=_worker, 
                                    args=(task_queue, done_queue)))
        if verbose:
            log.message("Starting {}".format(processes[i].name))
        processes[i].start()  
    
    for i in range(len(tasks)):
        rows, Dmp_part = done_queue.get()
        task_queue.put('STOP')
        if verbose:
            log.message("Merging submatrix {} (rows {}..{})".
                        format(i, rows[0], rows[-1]), flush=True)
        S_mp[rows, :] = Dmp_part
                    
    for p in processes:
        p.join()

    if verbose:
        log.message("Mirroring similarity matrix", flush=True)
    S_mp += S_mp.T

    if verbose:
        log.message("Setting self similarities", flush=True)        
    for i in range(n):
        S_mp[i, i] = self_value
        
    return S_mp.tocsr() 

def _partial_mp_emp_sparse(batch, matrix, idx, n, verbose):
    """Parallel helper function for MP empiric for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_empiric()
    """
    log = ConsoleLogging()
    S_mp = lil_matrix((len(batch), n), dtype=np.float32)
    
    # TODO implement faster version from serial MP emp sparse
    for i, b in enumerate(batch):
        if verbose and ((batch[i]+1)%1000 == 0 or batch[i] == n-1 
                        or i == len(batch)-1 or i == 0):
            log.message("MP_empiric_sparse: {} of {}. On {}.".format(
                batch[i]+1, n, mp.current_process().name), flush=True)
        for j in range(b+1, n):
            d = matrix[b, j]
            if d > 0: 
                dI = matrix.getrow(b).toarray()
                dJ = matrix.getrow(j).toarray()
                # non-zeros elements
                nz = (dI > 0) & (dJ > 0) 
                S_mp[i, j] = (nz & (dI <= d) & (dJ <= d)).sum() / (nz.sum() - 2)
                # need to mirror later
            else:
                pass # skip zero entries
    
    return (batch, S_mp)

def mutual_proximity_gaussi(D:np.ndarray, metric:str='distance', 
                            sample_size:int=0, test_set_ind:np.ndarray=None, 
                            verbose:int=0, n_jobs:int=-1, mv=None):
    """Transform a distance matrix with Mutual Proximity (indep. normal distr.).
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix. Gaussi 
    variant assumes independent normal distributions (FAST).
    The resulting second. distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray or csr_matrix
        - ndarray: The ``n x n`` symmetric distance or similarity matrix.
        - csr_matrix: The ``n x n`` symmetric similarity matrix.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.
        
        NOTE: In case of sparse `D`, only 'similarity' is supported.
    
    sample_size : int, optional (default: 0)
        Define sample size from which Gauss parameters are estimated.
        Use all data when set to ``0``.
    
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
    
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
    
    n_jobs : int, optional (default: -1)
        Number of parallel processes to be used.
        
        NOTE: set ``n_jobs=-1`` to use all CPUs
    
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
    n = D.shape[0]  # @UnusedVariable
    log = Logging.ConsoleLogging()
    log.warning("MP parallel code is not up-to-date! "
                "Please use methods of the MutualProximity module for now.")
    IO.check_distance_matrix_shape(D)
    IO.check_valid_metric_parameter(metric)
    # DO NOT DELETE comment
    #===========================================================================
    # # Checking input
    # if D.shape[0] != D.shape[1]:
    #     raise TypeError("Distance/similarity matrix is not quadratic.")        
    # if metric == 'similarity':
    #     self_value = 1
    # elif metric == 'distance':
    #     self_value = 0
    # else:
    #     raise ValueError("Parameter metric must be 'distance' or 'similarity'.") 
    #===========================================================================
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
     
    #===========================================================================
    # # Start MP Gaussi    
    # if verbose:
    #     log.message('Mutual Proximity Gaussi rescaling started.', flush=True)
    # D = D.copy()
    #===========================================================================

    if issparse(D):
        return _mutual_proximity_gaussi_sparse(D, sample_size, train_set_ind, 
                                               verbose, log, mv, n_jobs)
    else:
        log.warning("MP gaussi does not support parallel execution for dense "
                    "matrices at the moment. Continuing with 1 process.")
        from hub_toolbox.MutualProximity import mutual_proximity_gaussi
        return mutual_proximity_gaussi(D, metric, sample_size, test_set_ind, verbose)

def _mutual_proximity_gaussi_sparse(S, sample_size, train_set_ind, 
                                    verbose, log, mv, n_jobs):
    """MP gaussi for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gaussi()
    """
    n = S.shape[0]
    self_value = 1. # similarity matrix
    
    if mv is None:
        # mean, variance WITH zero values:
        from sklearn.utils.sparsefuncs_fast \
            import csr_mean_variance_axis0  # @UnresolvedImport
        mu, va = csr_mean_variance_axis0(S[train_set_ind])
    elif mv == 0:
        # mean, variance WITHOUT zero values (missing values)
        mu = np.array((S.sum(0) - 1) / (S.getnnz(0) - 1)).ravel()
        X = S.copy()
        X.data **= 2
        E1 = np.array((X.sum(0) - 1) / (X.getnnz(0) - 1)).ravel()
        del X
        va = E1 - mu**2
        del E1
    else:
        log.error("MP only supports missing values as zeros.", flush=True)
        raise ValueError("mv must be None or 0.")
    sd = np.sqrt(va)
    del va
            
    Dmp = lil_matrix(S.shape)
    
    # Parallelization
    if n_jobs == -1: # take all cpus
        NUMBER_OF_PROCESSES = mp.cpu_count()
    else:
        NUMBER_OF_PROCESSES = n_jobs
    tasks = []
    
    batches = _get_weighted_batches(n, NUMBER_OF_PROCESSES)
    
    for idx, batch in enumerate(batches):
        matrix = S
        tasks.append((_partial_mp_gaussi_sparse, 
                      (batch, matrix, idx, n, mu, sd, verbose)))   
    
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    for task in tasks:
        task_queue.put(task)
        
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        processes.append(mp.Process(target=_worker, 
                                    args=(task_queue, done_queue))) 
        processes[i].start()  
    
    for i in range(len(tasks)):  # @UnusedVariable
        rows, Dmp_part = done_queue.get()
        task_queue.put('STOP')
        if verbose:
            log.message("Merging submatrix {} (rows {}..{})".
                        format(i, rows[0], rows[-1]), flush=True)
        if rows.size > 0:
            rows_slice = slice(rows[0], rows[-1]+1)
        else:
            rows_slice = slice(0, 0)
        Dmp[rows_slice, :] = Dmp_part
     
    for p in processes:
        p.join()
    
    Dmp = Dmp.tolil()
    if verbose:
        log.message("Mirroring distance matrix", flush=True)
    Dmp += Dmp.T
    
    if verbose:
        log.message("Setting self distances", flush=True)
    for i in range(Dmp.shape[0]):
        Dmp[i, i] = self_value

    if verbose:
        log.message("Converting to CSR matrix", flush=True)
    return Dmp.tocsr()

def _partial_mp_gaussi_sparse(batch, matrix, idx, n, mu, sd, verbose):
    """Parallel helper function for MP gaussi for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gaussi()
    """
    log = ConsoleLogging()
    Dmp = lil_matrix((len(batch), n), dtype=np.float32)
    
    #non-vectorized code
    for i, b in enumerate(batch):
        if verbose and ((batch[i]+1)%1000 == 0 or batch[i]+1==n 
                        or i==len(batch)-1 or i==0):
            log.message("MP_gaussi_sparse: {} of {}. On {}.".format(
                batch[i]+1, n, mp.current_process().name, flush=True))
        for j in range(b+1, n):
            if matrix[b, j] > 0:       
                p1 = norm.cdf(matrix[b, j], mu[b], sd[b])
                p2 = norm.cdf(matrix[j, b], mu[j], sd[j])
                Dmp[i, j] = (p1 * p2).ravel()
                                 
    return batch, Dmp

def mutual_proximity_gammai(D:np.ndarray, metric:str='distance', 
                            test_set_ind:np.ndarray=None, verbose:int=0, 
                            n_jobs:int=-1, mv=None):
    """Transform a distance matrix with Mutual Proximity (indep. Gamma distr.).
    
    Applies Mutual Proximity (MP) [1]_ on a distance/similarity matrix. Gammai 
    variant assumes independent Gamma distributed distances (FAST).
    The resulting second. distance/similarity matrix should show lower hubness.
    
    Parameters
    ----------
    D : ndarray or csr_matrix
        - ndarray: The ``n x n`` symmetric distance or similarity matrix.
        - csr_matrix: The ``n x n`` symmetric similarity matrix.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.
        
        NOTE: In case of sparse `D`, only 'similarity' is supported.
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
        
    n_jobs : int, optional (default: -1)
        Number of parallel processes to be used.
        
        NOTE: set ``n_jobs=-1`` to use all CPUs
        
    Returns
    -------
    D_mp : ndarray
        Secondary distance MP gammai matrix.
    
    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
           Local and global scaling reduce hubs in space. The Journal of Machine 
           Learning Research, 13(1), 2871–2902.
    """
    log = Logging.ConsoleLogging()
    log.warning("MP parallel code is not up-to-date! "
                "Please use methods of the MutualProximity module for now.")
    IO.check_distance_matrix_shape(D)
    IO.check_valid_metric_parameter(metric)
    n = D.shape[0]
    sample_size = 0 # not implemented
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)    
    if issparse(D):
        return _mutual_proximity_gammai_sparse(D, sample_size, train_set_ind, 
                                               verbose, log, mv, n_jobs)
    else:
        log.warning("MP gammai does not support parallel execution for dense "
                    "matrices at the moment. Continuing with 1 process.")
        from hub_toolbox.MutualProximity import mutual_proximity_gammai
        return mutual_proximity_gammai(D, metric, test_set_ind, verbose)

def _mutual_proximity_gammai_sparse(S, sample_size=0, train_set_ind=None, 
                                    verbose=0, log=None, mv=None, n_jobs=-1):
    """MP gaussi for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gaussi()
    """
    self_value = 1. # similarity matrix
    # mean, variance WITHOUT zero values (missing values), ddof=1
    if S.diagonal().max() != self_value or S.diagonal().min() != self_value:
        raise ValueError("Self similarities must be 1.")
    S_param = S[train_set_ind]

    if mv is None:
        # mean, variance WITH zero values:
        from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  # @UnresolvedImport
        mu, va = csr_mean_variance_axis0(S_param)
    elif mv == 0: # mean, variance WITHOUT zero values (missing values)
        # the -1 accounts for self sim that must be excluded from the calc
        mu = np.array((S_param.sum(0) - 1) / (S_param.getnnz(0) - 1)).ravel()
        E2 = mu**2
        X = S_param.copy()
        X.data **= 2
        n_x = (X.getnnz(0) - 1)
        E1 = np.array((X.sum(0) - 1) / n_x).ravel()
        del X
        # for an unbiased sample variance
        va = n_x / (n_x - 1) * (E1 - E2)
        del E1
    else:
        log.error("MP only supports missing values as zeros.", flush=True)
        raise ValueError("mv must be None or 0.")
    
    A = (mu**2) / va
    B = va / mu
    del mu, va
    A[A < 0] = np.nan
    B[B <= 0] = np.nan

    S_mp = lil_matrix(S.shape, dtype=np.float32)
    n = S.shape[0]
    
    # Parallelization
    if n_jobs == -1: # take all cpus
        NUMBER_OF_PROCESSES = mp.cpu_count()
    else:
        NUMBER_OF_PROCESSES = n_jobs
    tasks = []
    
    batches = _get_weighted_batches(n, NUMBER_OF_PROCESSES)
    #   create jobs
    for idx, batch in enumerate(batches):
        matrix = S
        tasks.append((_partial_mp_gammai_sparse, 
                      (batch, matrix, idx, n, A, B, verbose)))   
    
    task_queue = mp.Queue()
    done_queue = mp.Queue()
    
    for task in tasks:
        task_queue.put(task)
    #   start jobs
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        processes.append(mp.Process(target=_worker, 
                                    args=(task_queue, done_queue)))
        processes[i].start()  
    #   collect results
    for i in range(len(tasks)):
        rows, Dmp_part = done_queue.get()
        task_queue.put('STOP')
        if verbose:
            log.message("Merging submatrix {} (rows {}..{})".
                        format(i, rows[0], rows[-1]), flush=True)
        if rows.size > 0:
            row_slice = slice(rows[0], rows[-1]+1)
        else: # for very small matrices, some batches might be empty
            row_slice = slice(0, 0)
        S_mp[row_slice] = Dmp_part
     
    for p in processes:
        p.join()
    
    S_mp = S_mp.tolil()
    if verbose:
        log.message("Mirroring distance matrix", flush=True)
    S_mp += S_mp.T
    
    if verbose:
        log.message("Setting self distances", flush=True)
    for i in range(S_mp.shape[0]):
        S_mp[i, i] = self_value

    if verbose:
        log.message("Converting to CSR matrix", flush=True)
    return S_mp.tocsr()

def _partial_mp_gammai_sparse(batch, matrix, idx, n, A, B, verbose):
    """Parallel helper function for MP gammai for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gammai()
    """
    log = ConsoleLogging()
    S_mp = lil_matrix((len(batch), n), dtype=np.float32)
    
    for i, b in enumerate(batch):
        if verbose and ((batch[i]+1)%1000 == 0 or batch[i]+1 == n 
                        or i == len(batch)-1 or i == 0):
            log.message("MP_gammai_sparse: {} of {}. On {}.".format(
                batch[i]+1, n, mp.current_process().name, flush=True))
        j_idx = slice(b+1, n)
        
        if b+1 >= n:
            continue # nothing to do in the last row
        #if j_idx.size == 0:
        #    continue 
        
        # avoiding fancy indexing for efficiency reasons
        S_ij = matrix[b, j_idx].toarray().ravel() #Extract dense rows temporarily        
        p1 = _local_gamcdf(S_ij, A[b], B[b])
        del S_ij
        
        S_ji = matrix[j_idx, b].toarray().ravel() #for vectorization below.
        p2 = _local_gamcdf(S_ji, A[j_idx], B[j_idx])
        del S_ji

        val = (p1 * p2).ravel()
        S_mp[i, j_idx] = val 
        #need to mirror later!!   
    
    return batch, S_mp

def _get_weighted_batches(n, jobs):
    """Define batches with increasing size to average the runtime for each
    batch.
    
    Observation: MP gets faster while processing the distance matrix, since it 
    processes half the matrix. This approximately follows a linear function.
    Idea: Give each row a weight w according to w(r) = 1 - r/n, where 
    r... row number, n... number of rows. Now each batch should get the same
    weight bw := n/jobs, where jobs... number of processes. The equation to 
    solve is then sum[r=a..b]( 1-r/n ) = n / 2jobs. 
    Solving for b yields the formula used below. (The last batch may get
    additional left-over rows).    
    """
     
    batches = []    
    a = 0
    b = 0
    for i in range(jobs-1):  # @UnusedVariable
        b = int((np.sqrt(jobs) * (2*n - 1)
                 - np.sqrt(jobs * (-2*a + 2*n + 1)**2 - 4*n**2))
                / (2*np.sqrt(jobs)))
        if b < 1:
            b = 1 # Each batch must contain at least 1 row
        batches.append(np.arange(a, b))
        a = b 
    if b < n-1:
        batches.append(np.arange(a, n))
    return batches

def _worker(work_input, work_output):
    """A helper function for cv parallelization."""
    for func, args in iter(work_input.get, 'STOP'):
        result = func(*args)
        work_output.put(result)

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

if __name__ == '__main__':
    """Test mp emp similarity sparse sequential & parallel implementations"""
    from scipy.sparse import rand, triu
    from hub_toolbox.Hubness import hubness
    from hub_toolbox.IO import load_dexter
    from hub_toolbox.KnnClassification import score
    #do = 'random'
    do = 'dexter'
    if do == 'random':
        print("RANDOM DATA:")
        print("------------")
        S = triu(rand(1000, 1000, 0.05, 'csr', np.float32, 43), 1)
        S += S.T
        D = 1. - S.toarray()
    elif do == 'dexter':
        print("DEXTER:")
        print("-------")
        D, c, v = load_dexter()
        acc_d, _, _ = score(D, c, [5], 'distance')
        S = csr_matrix(1 - D)
        acc_s, _, _ = score(S, c, [5], 'similarity')
   
    Sn_d, _, _ = hubness(D, 5, 'distance')
    Sn_s, _, _ = hubness(S, 5, 'similarity')
    print("Orig. dist. hubness:", Sn_d)
    print("Orig. sim.  hubness:", Sn_s)
    if do == 'dexter':
        print("Orig. dist. k-NN accuracy:", acc_d)
        print('Orig. sim.  k-NN accuracy:', acc_s)
        
    D_mp_emp_d = mutual_proximity_empiric(D)
    D_mp_emp_s = mutual_proximity_empiric(S, 'similarity')
    Sn_mp_emp_d, _, _ = hubness(D_mp_emp_d, 5)
    Sn_mp_emp_s, _, _ = hubness(D_mp_emp_s, 5, 'similarity')
    print("MP emp dist. hubness:", Sn_mp_emp_d)
    print("MP emp sim.  hubness:", Sn_mp_emp_s)
    if do == 'dexter':
        acc_mp_emp_d, _, _ = score(D_mp_emp_d, c, [5], 'distance')
        acc_mp_emp_s, _, _ = score(D_mp_emp_s, c, [5], 'similarity')
        print("MP emp dist. k-NN accuracy:", acc_mp_emp_d)
        print("MP emp sim.  k-NN accuracy:", acc_mp_emp_s)
        
    D_mp_gaussi_d = mutual_proximity_gaussi(D)
    D_mp_gaussi_s = mutual_proximity_gaussi(S, 'similarity')
    Sn_mp_gaussi_d, _, _ = hubness(D_mp_gaussi_d, 5)
    Sn_mp_gaussi_s, _, _ = hubness(D_mp_gaussi_s, 5, 'similarity')
    print("MP gaussi dist. hubness:", Sn_mp_gaussi_d)
    print("MP gaussi sim.  hubness:", Sn_mp_gaussi_s)
    if do == 'dexter':
        acc_mp_gaussi_d, _, _ = score(D_mp_gaussi_d, c, [5], 'distance')
        acc_mp_gaussi_s, _, _ = score(D_mp_gaussi_s, c, [5], 'similarity')
        print("MP gammai dist. k-NN accuracy:", acc_mp_gaussi_d)
        print("MP gammai sim.  k-NN accuracy:", acc_mp_gaussi_s)
    
    D_mp_gammai_d = mutual_proximity_gammai(D, 'distance')
    D_mp_gammai_s = mutual_proximity_gammai(S, 'similarity')
    Sn_mp_gammai_d, _, _ = hubness(D_mp_gammai_d, 5, 'distance')
    Sn_mp_gammai_s, _, _ = hubness(D_mp_gammai_s, 5, 'similarity')
    print("MP gammai dist. hubness:", Sn_mp_gammai_d)
    print("MP gammai sim.  hubness:", Sn_mp_gammai_s)
    if do == 'dexter':
        acc_mp_gammai_d, _, _ = score(D_mp_gammai_d, c, [5], 'distance')
        acc_mp_gammai_s, _, _ = score(D_mp_gammai_s, c, [5], 'similarity')
        print("MP gammai dist. k-NN accuracy:", acc_mp_gammai_d)
        print("MP gammai sim.  k-NN accuracy:", acc_mp_gammai_s)
    