"""
Computes the hubness of a distance matrix using its k nearest neighbors.
Hubness [1] is the skewness of the n-occurrence histogram.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

Usage:
  Sn = hubness(D) - Computes the hubness (Sk) of the n=5 occurrence histogram
     (standard)

  [Sn, Dk, Nk] hubness(D, k) - Computes the hubness of the n-occurrence
     histogram where n (k) is given. Nk is the n-occurrence histogram, Dk
     are the k nearest neighbors.

[1] Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data
Radovanovic, Nanopoulos, Ivanovic, Journal of Machine Learning Research 2010

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-17
"""

import numpy as np
from scipy import stats as stat
from scipy import sparse
from hub_toolbox import IO, Logging
from scipy.sparse.base import issparse
import multiprocessing as mp

class Hubness():
    """Computes the hubness of a distance matrix using its k nearest neighbors.
    
    Hubness is the skewness of the n-occurrence histogram.
    """
    
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
        
    def _worker(self, work_input, work_output):
        """A helper function for cv parallelization."""
        for func, args in iter(work_input.get, 'STOP'):
            result = self._calculate(func, args)
            work_output.put(result)
            
    def _calculate(self, func, args):
        """A helper function for cv parallelization."""
        return func(*args)
                
    def _partial_hubness(self, rows, submatrix, idx, n, verbose):
        """Parallel hubness calculation: Get k nearest neighbors for all points 
        in 'rows'"""
        
        Dk = np.zeros( (self.k, len(rows)), dtype=np.float32 )
        
        for i, row in enumerate(submatrix):
            if verbose and ((rows[i]+1)%10000==0 or rows[i]+1==n):
                self.log.message("NN: {} of {}.".format(rows[i]+1, n), flush=True)
            if issparse(submatrix):
                d = row.toarray().ravel() # dense copy of one row
            elif isinstance(submatrix, np.memmap):
                d = np.copy(row.astype(np.float32)) # in memory copy
            else: # normal ndarray
                d = row
            d[rows[i]] = self.d_self
            d[~np.isfinite(d)] = self.d_self
            # randomize the distance matrix rows to avoid the problem case
            # if all numbers to sort are the same, which would yield high
            # hubness, even if there is none
            rp = np.indices( (n, ) )[0]
            rp = np.random.permutation(rp)
            d2 = d[rp]
            d2idx = np.argsort(d2, axis=0)[::self.sort_order]
            Dk[:, i] = rp[d2idx[0:self.k]]  
        
        return [rows, Dk]    
    
    
    def calculate_hubness(self, debug=False, n_jobs=-1):
        """Calculate hubness."""
                
        verbose = debug
        if verbose:
            self.log.message("Start hubness calculation "
                             "(skewness of {}-occurence)".format(self.k))
            
        # Initialization
        n = self.D.shape[0]
        Dk = np.zeros( (self.k, np.size(self.D, 1)), dtype=np.float32 )
        
        if not isinstance(self.D, np.memmap) and \
            not sparse.issparse(self.D): 
            # correct self-distance must be ensured upstream for sparse/memmap
            # Set self dist to inf
            np.fill_diagonal(self.D, self.d_self)
            # make non-finite (NaN, Inf) appear at the end of the sorted list
            self.D[~np.isfinite(self.D)] = self.d_self
                            
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
        #if batches[-1][-1] < n-1:
        batches.append( np.arange((NUMBER_OF_PROCESSES-1)*batch_size, n) )
        
        for idx, batch in enumerate(batches):
            submatrix = self.D[batch]
            tasks.append((self._partial_hubness, (batch, submatrix, idx, n, verbose)))   
        
        task_queue = mp.Queue()  # @UndefinedVariable
        done_queue = mp.Queue()    # @UndefinedVariable
        
        for task in tasks:
            task_queue.put(task)
            
        for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
            mp.Process(target=self._worker, args=(task_queue, done_queue)).start()  # @UndefinedVariable
        
        for i in range(len(tasks)):  # @UnusedVariable
            rows, Dk_part = done_queue.get()
            Dk[:, rows] = Dk_part[:, :]
            
        for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
            task_queue.put('STOP')        
                   
        # N-occurence
        Nk = np.bincount(Dk.astype(int).ravel())    
        # Hubness
        Sn = stat.skew(Nk)
         
        if verbose:
            self.log.message("Hubness calculation done.", flush=True)
            
        # return hubness, k-nearest neighbors, N occurence
        return (Sn, Dk, Nk)
    
if __name__ == '__main__':
    from hub_toolbox import HubnessAnalysis as ha
    h = ha.HubnessAnalysis()
    #D, c, v = h.load_dexter()
    hub = Hubness(h.D)
    Sn, Dk, Nk = hub.calculate_hubness(True, n_jobs=-1)
    print("Hubness =", Sn)