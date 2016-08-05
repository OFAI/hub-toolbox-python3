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
from scipy import stats
from hub_toolbox import IO, Logging
from scipy.sparse.base import issparse
import multiprocessing as mp
import sys

def hubness(D:np.ndarray, k:int=5, metric='distance', 
            verbose:int=0, n_jobs:int=-1):
    """Compute hubness of a distance matrix.
    
    Hubness [1] is the skewness of the k-occurrence histogram (reverse nearest 
    neighbor count, i.e. how often does a point occur in the k-nearest 
    neighbor lists of other points).
    
    Parameters:
    -----------
    D : ndarray
        The n x n symmetric distance (similarity) matrix.
    
    k : int, optional (default: 5)
        Neighborhood size for k-occurence.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix
    
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
        
    n_jobs : int, optional (default: -1)
        Number of parallel processes spawned for hubness calculation.
        Default value (-1): number of available CPUs.
        
    Returns:
    --------
    S_k : float
        Hubness (skewness of k-occurence distribution)
    D_k : ndarray
        k-nearest neighbor lists
    N_k : ndarray
        k-occurence list    
    
    See also:
    ---------
    [1] Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010). 
    Hubs in Space : Popular Nearest Neighbors in High-Dimensional Data. 
    Journal of Machine Learning Research, 11, 2487–2531. Retrieved from 
    http://jmlr.csail.mit.edu/papers/volume11/radovanovic10a/radovanovic10a.pdf
    """
    log = Logging.ConsoleLogging()
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
    elif metric == 'similarity':
        d_self = -np.inf
        sort_order = -1
    else:
        raise ValueError("Parameter 'metric' must be 'distance' or "
                         "'similarity'.")
        
    if verbose:
        log.message("Hubness calculation (skewness of {}-occurence)".format(k))
        
    # Initialization
    n = D.shape[0]
    D = D.copy()
    D_k = np.zeros((k, D.shape[1]), dtype=np.float32 )
    
    if issparse(D): 
        pass # correct self-distance must be ensured upstream for sparse
    else:
        # Set self dist to inf
        np.fill_diagonal(D, d_self)
        # make non-finite (NaN, Inf) appear at the end of the sorted list
        D[~np.isfinite(D)] = d_self
                        
    # Parallelization
    if n_jobs == -1: # take all cpus
        NUMBER_OF_PROCESSES = mp.cpu_count()  # @UndefinedVariable
    else:
        NUMBER_OF_PROCESSES = n_jobs
    tasks = []
    
    batches = []
    batch_size = n // NUMBER_OF_PROCESSES
    for i in range(NUMBER_OF_PROCESSES-1):
        batches.append( np.arange(i*batch_size, (i+1)*batch_size) )
    batches.append( np.arange((NUMBER_OF_PROCESSES-1)*batch_size, n) )
    
    for idx, batch in enumerate(batches):
        submatrix = D[batch[0]:batch[-1]+1]
        tasks.append((_partial_hubness, 
                     (k, d_self, log, sort_order, 
                      batch, submatrix, idx, n, verbose)))   
    
    task_queue = mp.Queue()  # @UndefinedVariable
    done_queue = mp.Queue()  # @UndefinedVariable
    
    for task in tasks:
        task_queue.put(task)
        
    for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
        mp.Process(target=_worker, args=(task_queue, done_queue)).start()  # @UndefinedVariable
    
    for i in range(len(tasks)):  # @UnusedVariable
        rows, Dk_part = done_queue.get()
        D_k[:, rows[0]:rows[-1]+1] = Dk_part
        
    for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
        task_queue.put('STOP')        
               
    # k-occurence
    N_k = np.bincount(D_k.astype(int).ravel())    
    # Hubness
    S_k = stats.skew(N_k)
     
    if verbose:
        log.message("Hubness calculation done.", flush=True)
        
    # return hubness, k-nearest neighbors, N occurence
    return S_k, D_k, N_k
        
def _worker(work_input, work_output):
    """A helper function for cv parallelization."""
    for func, args in iter(work_input.get, 'STOP'):
        result = _calculate(func, args)
        work_output.put(result)
            
def _calculate(func, args):
    """A helper function for cv parallelization."""
    return func(*args)
            
def _partial_hubness(k, d_self, log, sort_order, 
                     rows, submatrix, idx, n, verbose):
    """Parallel hubness calculation: Get k nearest neighbors for all points 
    in 'rows'"""
    
    Dk = np.zeros((k, len(rows)), dtype=np.float32)
    
    for i, row in enumerate(submatrix):
        if verbose and ((rows[i]+1)%10000==0 or rows[i]+1==n):
            log.message("NN: {} of {}.".format(rows[i]+1, n), flush=True)
        if issparse(submatrix):
            d = row.toarray().ravel() # dense copy of one row
        else: # normal ndarray
            d = row
        d[rows[i]] = d_self
        d[~np.isfinite(d)] = d_self
        # randomize the distance matrix rows to avoid the problem case
        # if all numbers to sort are the same, which would yield high
        # hubness, even if there is none
        rp = np.random.permutation(n)
        d2 = d[rp]
        d2idx = np.argsort(d2, axis=0)[::sort_order]
        Dk[:, i] = rp[d2idx[0:k]]  
    
    return [rows, Dk]    


class Hubness():
    """DEPRECATED"""
    
    def __init__(self, D, k:int=5, isSimilarityMatrix:bool=False):
        self.log = Logging.ConsoleLogging()
        if isinstance(D, np.memmap):
            self.D = D
        else:
            self.D = IO.copy_D_or_load_memmap(D, writeable=False)
        self.k = k
        if isSimilarityMatrix:
            self.d_self = -np.inf
            self.sort_order = -1 # descending, interested in highest similarity
        else:
            self.d_self = np.inf
            self.sort_order = 1 # ascending, interested in smallest distance
        np.random.seed()
        
    def calculate_hubness(self, debug=False, n_jobs=-1):
        """Calculate hubness."""
        print("DEPRECATED: Please use Hubness_parallel.hubness().", 
              file=sys.stderr)
        if self.sort_order == 1:
            metric = 'distance'
        elif self.sort_order == -1:
            metric = 'similarity'
        else:
            raise ValueError("sort_order must be -1 or 1.")
        
        return hubness(self.D, self.k, metric, debug, n_jobs)   
        
if __name__ == '__main__':
    from hub_toolbox import HubnessAnalysis as ha
    h = ha.HubnessAnalysis()
    hub = Hubness(h.D)
    Sn, Dk, Nk = hub.calculate_hubness(True, n_jobs=-1)
    print("Hubness =", Sn)