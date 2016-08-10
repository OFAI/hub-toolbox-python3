#!/usr/bin/env python
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
"""

import numpy as np
from scipy.special import gammainc  # @UnresolvedImport
from scipy.stats import norm
from scipy.sparse import issparse, lil_matrix
from enum import Enum
import multiprocessing as mp
from hub_toolbox import IO, Logging
from hub_toolbox.Logging import FileLogging

def mutual_proximity_empiric(D:np.ndarray, metric:str='distance', 
                             test_set_ind:np.ndarray=None, verbose:int=0,
                             n_jobs:int=-1):
    """Transform a distance matrix with Mutual Proximity (empiric distribution).
    
    Applies Mutual Proximity (MP) [1] on a distance/similarity matrix using 
    the empiric data distribution (EXACT, rather SLOW). The resulting 
    secondary distance/similarity matrix should show lower hubness.
    
    Parameters:
    -----------
    D : ndarray or csr_matrix
        - ndarray: The n x n symmetric distance or similarity matrix.
        - csr_matrix: The n x n symmetric similarity matrix.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix.
        NOTE: In case of sparse D, only 'similarity' is supported.
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
        
    n_jobs : int, optional (default: -1)
        Number of parallel processes to be used.
        NOTE: set n_jobs=-1 to use all CPUs
        
    Returns:
    --------
    D_mp : ndarray
        Secondary distance MP empiric matrix.
    
    See also:
    ---------
    [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
    Local and global scaling reduce hubs in space. The Journal of Machine 
    Learning Research, 13(1), 2871–2902.
    """
    log = Logging.ConsoleLogging()
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
        matrix = D
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
    log = FileLogging()
    S_mp = lil_matrix((len(batch), n), dtype=np.float32)
    
    # TODO implement faster version from serial MP emp sparse
    for i, b in enumerate(batch):
        if verbose and ((batch[i]+1)%1000 == 0 or batch[i]==n-1 
                        or i==len(batch)-1 or i==0):
            log.message("MP_empiric_sparse: {} of {}. On {}.".format(
                        batch[i]+1, n, mp.current_process().name), flush=True)
        for j in range(b+1, n):
            d = matrix[b, j]
            if d>0: 
                dI = D[b, :].toarray()
                dJ = D[j, :].toarray()
                # non-zeros elements
                nz = (dI > 0) & (dJ > 0) 
                S_mp[i, j] = (nz & (dI <= d) & (dJ <= d)).sum() / nz.sum()
                # need to mirror later
            else:
                pass # skip zero entries
    
    return (batch, S_mp)

def mutual_proximity_gaussi(D:np.ndarray, metric:str='distance', 
                             sample_size:int=0, test_set_ind:np.ndarray=None, 
                             verbose:int=0, n_jobs:int=-1, mv=None):
    """Transform a distance matrix with Mutual Proximity (indep. normal distr.).
    
    Applies Mutual Proximity (MP) [1] on a distance/similarity matrix. Gaussi 
    variant assumes independent normal distributions (FAST).
    The resulting second. distance/similarity matrix should show lower hubness.
    
    Parameters:
    -----------
    D : ndarray or csr_matrix
        - ndarray: The n x n symmetric distance or similarity matrix.
        - csr_matrix: The n x n symmetric similarity matrix.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix.
        NOTE: In case of sparse D, only 'similarity' is supported.
        
    sample_size : int, optional (default: 0)
        Define sample size from which Gauss parameters are estimated.
        Use all data when set to 0.
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
    
    n_jobs : int, optional (default: -1)
        Number of parallel processes to be used.
        NOTE: set n_jobs=-1 to use all CPUs
    
    Returns:
    --------
    D_mp : ndarray
        Secondary distance MP gaussi matrix.
    
    See also:
    ---------
    [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
    Local and global scaling reduce hubs in space. The Journal of Machine 
    Learning Research, 13(1), 2871–2902.
    """    
    # Initialization   
    n = D.shape[0]  # @UnusedVariable
    log = Logging.ConsoleLogging()
    
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
        mu = np.array(S.sum(0) / S.getnnz(0)).ravel()
        X = S.copy()
        X.data **= 2
        E1 = np.array(X.sum(0) / X.getnnz(0)).ravel()
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
        Dmp[rows, :] = Dmp_part
     
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
    log = FileLogging()
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
        batches.append( np.arange(a, b) )
        a = b 
    if b < n-1:
        batches.append( np.arange(a, n) )
    return batches

def _worker(work_input, work_output):
    """A helper function for cv parallelization."""
    for func, args in iter(work_input.get, 'STOP'):
        result = func(*args)
        work_output.put(result)
        
def _local_gamcdf(x, a, b, mv=np.nan):
    """Gamma CDF, moment estimator"""
    try:
        a[a<0] = np.nan
    except TypeError:
        if a < 0:
            a = np.nan
    try:
        b[b<=0] = np.nan
    except TypeError:
        if b <= 0:
            b = np.nan
    x[x<0] = 0
    
    # don't calculate gamcdf for missing values
    if mv == 0:
        nz = x>0
        z = x[nz] / b[nz]
        p = np.zeros_like(x)
        p[nz] = gammainc(a[nz], z)
    else:
        z = x / b
        p = gammainc(a, z)
    return p
    
##############################################################################
#
# DEPRECATED class
#
class MutualProximity():
    """DEPRECATED"""
    
    def __init__(self, D, isSimilarityMatrix=False, missing_values=None, tmp='/tmp/'):
        """DEPRECATED"""
        self.D = IO.copy_D_or_load_memmap(D, writeable=True)
        self.log = Logging.ConsoleLogging()
        if isSimilarityMatrix:
            self.self_value = 1
        else:
            self.self_value = 0
        self.isSimilarityMatrix = isSimilarityMatrix
        self.tmp = tmp
        if missing_values is None:
            if issparse(D):
                self.mv = 0
            else:
                self.mv = None
        else: 
            self.mv = missing_values
        
    def calculate_mutual_proximity(self, distrType=None, test_set_mask=None, 
                                   verbose=False, sample_size=0, empspex=False, 
                                   n_jobs=-1):
        """DEPRECATED"""
        
        if verbose:
            self.log.message('Mutual proximity rescaling started.', flush=True)
            
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)
            
        if distrType is None:
            self.log.message("No Mutual Proximity type given. "
                             "Using: Distribution.empiric. "
                             "For fast results use: Distribution.gaussi")
            Dmp = self.mp_empiric(train_set_mask, verbose, empspex, n_jobs)
        else:
            if distrType == Distribution.empiric:
                Dmp = self.mp_empiric(train_set_mask, verbose, empspex, n_jobs)
            elif distrType == Distribution.gaussi:
                Dmp = self.mp_gaussi(train_set_mask, verbose, sample_size, n_jobs)
            elif distrType == Distribution.gammai:
                Dmp = self.mp_gammai(train_set_mask, verbose, n_jobs)
            else:
                self.log.warning("Valid Mutual Proximity type missing!\n"+\
                             "Use: \n"+\
                             "mp = MutualProximity(D, Distribution.empiric|"+\
                             "Distribution.gaussi|"+\
                             "Distribution.gammi)\n"+\
                             "Dmp = mp.calculate_mutual_proximity()")
                Dmp = np.array([])
       
        return Dmp
                           
    def mp_empiric(self, train_set_mask=None, verbose=False, empspex=False, n_jobs=-1):
        """DEPRECATED"""   
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance' 
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        return mutual_proximity_empiric(self.D, metric, test_set_ind, verbose)
    
    def mp_gaussi(self, train_set_mask=None, verbose=False, 
                  sample_size=0, n_jobs=-1):
        """DEPRECATED"""
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance' 
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        return mutual_proximity_gaussi(D, metric, sample_size, test_set_ind, verbose, n_jobs)
        
    def _partial_mp_gammai_sparse(self, batch, matrix, idx, n, A, B, verbose):
        Dmp = lil_matrix((len(batch), self.D.shape[1]), dtype=np.float32)
        
        for i, b in enumerate(batch):
            if verbose and ((batch[i]+1)%1000 == 0 or batch[i]+1==n or i==len(batch)-1 or i==0):
                self.log.message("MP_gammai_sparse: {} of {}. On {}.".format(
                                batch[i]+1, n, mp.current_process().name, flush=True))  # @UndefinedVariable
            j_idx = np.arange(b+1, n)
            j_len = np.size(j_idx)
             
            if j_idx.size == 0:
                continue # nothing to do in the last row
            
            # avoiding fancy indexing for efficiency reasons
            Dij = matrix[b, j_idx[0]:j_idx[-1]+1].toarray().ravel() #Extract dense rows temporarily
            Dji = matrix[j_idx[0]:j_idx[-1]+1, b].toarray().ravel() #for vectorization below.
            
            p1 = self.local_gamcdf(Dij, \
                                   np.tile(A[b], j_len), #(1, j_len)), \
                                   np.tile(B[b], j_len)) #(1, j_len)))
            del Dij
            
            p2 = self.local_gamcdf(Dji, 
                                   A[j_idx], 
                                   B[j_idx])
            del Dji

            val = (p1 * p2).ravel()
            Dmp[i, j_idx] = val 
            #need to mirror later!!   
        
        return batch, Dmp

    def mp_gammai_sparse(self, train_set_mask, verbose, n_jobs=-1):
        # TODO implement train_test split
        
        if self.mv is None:
            # mean, variance WITH zero values:
            from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  # @UnresolvedImport
            mu, va = csr_mean_variance_axis0(self.D[train_set_mask])
        elif self.mv == 0:
            # mean, variance WITHOUT zero values (missing values)
            mu = np.array(self.D.sum(0) / self.D.getnnz(0)).ravel()
            X = self.D.copy()
            X.data **= 2
            E1 = np.array(X.sum(0) / X.getnnz(0)).ravel()
            del X
            va = E1 - mu**2
            del E1
        else:
            self.log.error("MP only supports missing values as zeros. Aborting.")
            return
        
        A = (mu**2) / va
        B = va / mu
        del mu, va
        A[A<0] = np.nan
        B[B<=0] = np.nan

        Dmp = lil_matrix(self.D.shape, dtype=np.float32)
        n = self.D.shape[0]
        
        # Parallelization
        if n_jobs == -1: # take all cpus
            NUMBER_OF_PROCESSES = mp.cpu_count()  # @UndefinedVariable
        else:
            NUMBER_OF_PROCESSES = n_jobs
        tasks = []
        
        batches = _get_weighted_batches(n, NUMBER_OF_PROCESSES)
        
        for idx, batch in enumerate(batches):
            matrix = self.D#.copy() # no copy necessary!
            tasks.append((self._partial_mp_gammai_sparse, (batch, matrix, idx, n, A, B, verbose)))   
        
        task_queue = mp.Queue()     # @UndefinedVariable
        done_queue = mp.Queue()     # @UndefinedVariable
        
        for task in tasks:
            task_queue.put(task)
            
        processes = []
        for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
            processes.append(mp.Process(target=_worker, args=(task_queue, done_queue))) # @UndefinedVariable
            processes[i].start()  
        
        for i in range(len(tasks)):  # @UnusedVariable
            rows, Dmp_part = done_queue.get()
            task_queue.put('STOP')
            if verbose:
                self.log.message("Merging submatrix {} (rows {}..{})".format(i, rows[0], rows[-1]), flush=True)
            Dmp[rows, :] = Dmp_part
         
        for p in processes:
            p.join()
        
        Dmp = Dmp.tolil()
        if verbose:
            self.log.message("Mirroring distance matrix", flush=True)
        Dmp += Dmp.T
        
        if verbose:
            self.log.message("Setting self distances", flush=True)
        for i in range(Dmp.shape[0]):
            Dmp[i, i] = self.self_value

        if verbose:
            self.log.message("Converting to CSR matrix", flush=True)
        return Dmp.tocsr()
    
    def mp_gammai(self, train_set_mask=None, verbose=False, n_jobs=-1):
        """Compute Mutual Proximity modeled with independent Gamma distributions."""
        
        if not issparse(self.D):
            np.fill_diagonal(self.D, self.self_value)
        else:
            if self.isSimilarityMatrix:
                return self.mp_gammai_sparse(train_set_mask, verbose, n_jobs=n_jobs)
            else:
                self.log.error('MP gammai sparse only support similarities atm.'
                               ' Found distance matrix. Aborting.')
                return
            
        self.log.warning("MP gammai does not support parallel execution for "
                         "dense matrices at the moment. Continuing with 1 process.")
        if self.mv is not None:
            self.log.warning("MP gammai does not support missing value handling"
                             "for dense matrices atm.")
        mu = np.mean(self.D[train_set_mask], 0)
        va = np.var(self.D[train_set_mask], 0, ddof=1)
        A = (mu**2) / va
        B = va / mu
        
        Dmp = np.zeros_like(self.D)
        n = np.size(self.D, 0)
        
        for i in range(n):
            if verbose and ((i+1)%1000 == 0 or i+1==n):
                self.log.message("MP_gammai: {} of {}".format(i+1, n), flush=True)
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx)
            
            if self.isSimilarityMatrix:
                p1 = self.local_gamcdf(self.D[i, j_idx], \
                                       np.tile(A[i], (1, j_len)), \
                                       np.tile(B[i], (1, j_len)))
                p2 = self.local_gamcdf(self.D[j_idx, i].T, 
                                       A[j_idx], 
                                       B[j_idx])
                Dmp[i, i] = self.self_value
                Dmp[i, j_idx] = (p1 * p2).ravel()
            else:
                p1 = 1 - self.local_gamcdf(self.D[i, j_idx], \
                                           np.tile(A[i], (1, j_len)), \
                                           np.tile(B[i], (1, j_len)))
                p2 = 1 - self.local_gamcdf(self.D[j_idx, i].T, 
                                           A[j_idx], 
                                           B[j_idx])
                Dmp[i, j_idx] = (1 - p1 * p2).ravel()
                
            Dmp[j_idx, i] = Dmp[i, j_idx]               
        
        return Dmp   
    
class Distribution(Enum):
    """DEPRECATED"""
    empiric = 'empiric'
    gaussi = 'gaussi'
    gammai = 'gammai'
    
if __name__ == '__main__':
    """Test mp empiric similarity sparse sequential & parallel implementations"""
    from scipy.sparse import rand, csr_matrix
    #do = 'random'
    do = 'dexter'
    if do == 'random':
        D = rand(5000, 5000, 0.05, 'csr', np.float32, 42)
        D = np.triu(D.toarray())
        D = D + D.T
        np.fill_diagonal(D, 1)
        D = csr_matrix(D)
    elif do == 'dexter':
        from hub_toolbox import HubnessAnalysis, KnnClassification
        D, c, v = HubnessAnalysis.HubnessAnalysis().load_dexter()
        D = 1 - D
        D = csr_matrix(D)
        k = KnnClassification.KnnClassification(D, c, 5, True)
        acc, corr, cmat = k.perform_knn_classification()
        print('\nk-NN accuracy:', acc)
    from hub_toolbox import Hubness 
    #from hub_toolbox import MutualProximity as MutProx
    h = Hubness.Hubness(D, 5, True)
    Sn, _, _ = h.calculate_hubness()
    print("Hubness:", Sn)
    
    #===========================================================================
    # mp1 = MutProx.MutualProximity(D, True)
    # Dmp1 = mp1.calculate_mutual_proximity(MutProx.Distribution.gammai, None, True, False, 0, None, empspex=False)
    # h = Hubness.Hubness(Dmp1, 5, True)
    # Sn, _, _ = h.calculate_hubness()
    # print("Hubness (sequential):", Sn)
    #===========================================================================
    
    mp2 = MutualProximity(D, isSimilarityMatrix=True, missing_values=0)
    Dmp2 = mp2.calculate_mutual_proximity(Distribution.gammai, None, True, 0, empspex=False, n_jobs=4)
    h = Hubness.Hubness(Dmp2, 5, isSimilarityMatrix=True)
    Sn, _, _ = h.calculate_hubness()
    if do == 'dexter':
        k = KnnClassification.KnnClassification(Dmp2, c, 5, True)
        acc, corr, cmat = k.perform_knn_classification()
        print('\nk-NN accuracy:', acc)
    print("Hubness (parallel):", Sn)
    print(Dmp2.max(), Dmp2.min(), Dmp2.mean())
    
    #print("Summed differences in the scaled matrices:", (Dmp1-Dmp2).sum())
    #print("D", D)
    #print("Dmp1", Dmp1)
    #print("Dmp2", Dmp2)
    