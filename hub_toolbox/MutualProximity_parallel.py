"""
Applies Mutual Proximity (MP) [1] on a distance matrix. The return value is
converted to a distance matrix again. The resulting distance matrix
should show lower hubness.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

Usage:
  Dmp = mutual_proximity(D, type) - Applies MP on the distance matrix 'D'
     using the selected variant ('type'). The transformed distance matrix
     is returned.

Possible types:
  'empiric': Uses the Empirical distribution to perform Mutual Proximity.
  'gauss': (requires the Statistics Toolbox (the mvncdf() function)
     Assumes that the distances are Gaussian distributed.
  'gaussi': Assumes that the distances are independently Gaussian
     distributed. (fastest Variante)
  'gammai': Assumes that the distances follow a Gamma distribution and
     are independently distributed.

[1] Local and global scaling reduce hubs in space, 
Schnitzer, Flexer, Schedl, Widmer, Journal of Machine Learning Research 2012

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-25
"""

import numpy as np
from scipy.special import gammainc
from scipy.stats import norm, mvn
from scipy.sparse import issparse, lil_matrix
from enum import Enum
import multiprocessing as mp
from hub_toolbox import IO, Logging

class Distribution(Enum):
    empiric = 'empiric'
    gaussi = 'gaussi'
    gammai = 'gammai'

def _get_weighted_batches(n, jobs):
    """Define batches with increasing size to average the runtime for each
    batch.
    
    Observation: MP gets faster while processing the distance matrix. This 
    approximately follows a linear function.
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
        b = int((np.sqrt(jobs)*(2*n-1)-np.sqrt(jobs*(-2*a+2*n+1)**2-4*n**2))/(2*np.sqrt(jobs)))
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
        
class MutualProximity():
    """Transform a distance matrix with Mutual Proximity.
    
    """
    
    def __init__(self, D, isSimilarityMatrix=False, missing_values=None, tmp='/tmp/'):
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
        """Apply MP on a distance matrix."""
        
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
         
    def _partial_mp_empiric_sparse_exact(self, batch, matrix, idx, n, verbose):
        Dmp = np.zeros((len(batch), self.D.shape[1]), dtype=np.float32)
        for i, b in enumerate(batch):
            if verbose and ((batch[i]+1)%1000 == 0 or batch[i]==n-1 or i==len(batch)-1 or i==0):
                self.log.message("MP_empiric_sparse_exact: {} of {}. On {}."
                                 .format(batch[i]+1, n, mp.current_process().name), flush=True)  # @UndefinedVariable
            for j in range(b+1, n):
                d = matrix[b, j]
                dI = matrix[b, :].todense()
                dJ = matrix[j, :].todense()
                 
                if self.isSimilarityMatrix:
                    sIJ_intersect = ((dI <= d) & (dJ <= d)).sum()
                    sIJ_overlap = sIJ_intersect / n
                else:
                    sIJ_intersect = ((dI > d) & (dJ > d)).sum()
                    sIJ_overlap = 1 - (sIJ_intersect / n)
                Dmp[i, j] = sIJ_overlap
                #need to mirror later!!
        filenamewords = [str(batch[0]), str(batch[-1]), 'triu']
        f = self.tmp + '_'.join(filenamewords)
        self.log.message("MP_empiric_sparse_exact: Saving batch {}-{} to {}. On {}."
                        .format(batch[0], batch[-1], f, mp.current_process().name), flush=True)  # @UndefinedVariable
        np.save(f, Dmp)
        return (batch, Dmp)
    
    def _partial_mp_empiric_sparse(self, batch, matrix, idx, n, verbose):
        Dmp = lil_matrix((len(batch), self.D.shape[1]), dtype=np.float32)
        for i, b in enumerate(batch):
            if verbose and ((batch[i]+1)%1000 == 0 or batch[i]==n-1 or i==len(batch)-1 or i==0):
                self.log.message("MP_empiric_sparse: {} of {}. On {}."
                                 .format(batch[i]+1, n, mp.current_process().name), flush=True)  # @UndefinedVariable
            for j in range(b+1, n):
                d = matrix[b, j]
                if d>0: 
                    #nnz = np.max([self.D[b].nnz, self.D[j].nnz])
                    dI = self.D[b, :].todense()
                    dJ = self.D[j, :].todense()
                    # non-zeros elements
                    nz = (dI > 0) | (dJ > 0) # logical AND or OR here?!
                    # number of non-zero elements
                    nnz = nz.sum()
                    
                    if self.isSimilarityMatrix:
                        sIJ_intersect = (nz & (dI <= d) & (dJ <= d)).sum()
                        sIJ_overlap = sIJ_intersect / nnz
                    else:
                        sIJ_intersect = (nz & (dI > d) & (dJ > d)).sum()
                        sIJ_overlap = 1 - (sIJ_intersect / nnz)
                    Dmp[i, j] = sIJ_overlap
                    # need to mirror later
                else:
                    pass # skip zero entries
        
        #=======================================================================
        # filenamewords = [str(batch[0]), str(batch[-1]), 'triu']
        # f = self.tmp + '_'.join(filenamewords)
        # self.log.message("MP_empiric_sparse: Saving batch {}-{} to {}. On {}."
        #                 .format(batch[0], batch[-1], f, mp.current_process().name), flush=True)  # @UndefinedVariable
        # np.save(f, Dmp)
        #=======================================================================
        return (batch, Dmp)
    
    def mp_empiric_sparse(self, train_set_mask=None, verbose=False, empspex=False, n_jobs=-1):
        n = np.shape(self.D)[0]
        
        if empspex:
            Dmp = np.zeros(self.D.shape, dtype=np.float32)
        else:
            Dmp = lil_matrix(self.D.shape)
            
        # Parallelization
        if n_jobs == -1: # take all cpus
            NUMBER_OF_PROCESSES = mp.cpu_count()  # @UndefinedVariable
        else:
            NUMBER_OF_PROCESSES = n_jobs
        tasks = []
        
        batches = _get_weighted_batches(n, NUMBER_OF_PROCESSES)
        
        for idx, batch in enumerate(batches):
            matrix = self.D#.copy()
            if empspex:
                tasks.append((self._partial_mp_empiric_sparse_exact, (batch, matrix, idx, n, verbose)))   
            else:
                tasks.append((self._partial_mp_empiric_sparse, (batch, matrix, idx, n, verbose)))
        
        task_queue = mp.Queue()     # @UndefinedVariable
        done_queue = mp.Queue()     # @UndefinedVariable
        
        for task in tasks:
            task_queue.put(task)
            
        processes = []
        for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
            processes.append(mp.Process(target=_worker, args=(task_queue, done_queue))) # @UndefinedVariable
            if verbose:
                self.log.message("Starting {}".format(processes[i].name))
            processes[i].start()  
        
        for i in range(len(tasks)):  # @UnusedVariable
            rows, Dmp_part = done_queue.get()
            task_queue.put('STOP')
            if verbose:
                self.log.message("Merging submatrix {} (rows {}..{})".format(i, rows[0], rows[-1]), flush=True)
            Dmp[rows, :] = Dmp_part
                        
        for p in processes:
            p.join()

        #=======================================================================
        # import os, datetime
        # time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
        # folder = self.tmp + 'feldbauer/hubPDBs2/'
        # if empspex:
        #     filename = 'D_mpempspex_triu_'+time
        # else:
        #     filename = 'D_mpemp_triu_'+time
        # os.makedirs(folder, exist_ok=True)
        # if verbose:
        #     self.log.message("Saving Dmp upper to {} as {}.npy".format(folder, filename), flush=True)
        # np.save(folder + filename, Dmp)
        #=======================================================================
        
        if verbose:
            self.log.message("Mirroring distance matrix", flush=True)
        Dmp += Dmp.T
    
        if self.isSimilarityMatrix:
            if verbose:
                self.log.message("Setting self distances", flush=True)
                
            if empspex:
                np.fill_diagonal(Dmp, self.self_value) #need to set self values
            else:
                for i in range(n):
                    Dmp[i, i] = self.self_value
            
        if empspex:
            return Dmp
        else:
            return Dmp.tocsr()
                    
    def mp_empiric(self, train_set_mask=None, verbose=False, empspex=False, n_jobs=-1):
        """Compute Mutual Proximity distances with empirical data (slow)."""   
        #TODO implement train_set_mask!
        
        if not np.all(train_set_mask):
            self.log.error("MP empiric does not support train/test splits yet.")
            return
        
        if issparse(self.D):
            return self.mp_empiric_sparse(train_set_mask, verbose, empspex, n_jobs)
        else:
            # ensure correct self distances (NOT done for sparse matrices!)
            np.fill_diagonal(self.D, self.self_value)
            self.log.warning("MP empiric does not support parallel execution "
                             "for dense matrices at the moment. "
                             "Continuing with 1 process.")
        n = np.shape(self.D)[0]
        Dmp_list = [0 for i in range(n)]
         
        for i in range(n-1):
            if verbose and ((i+1)%1000 == 0 or i==n):
                self.log.message("MP_empiric: {} of {}.".format(i+1, n-1), flush=True)
            # Select only finite distances for MP
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx, 0)
             
            dI = np.tile(self.D[i, :], (j_len, 1))
            dJ = self.D[j_idx, :]
            d = np.tile(self.D[j_idx, i][:, np.newaxis], (1, n))
             
            if self.isSimilarityMatrix:
                sIJ_intersect = np.sum((dI <= d) & (dJ <= d), 1)
                sIJ_overlap = sIJ_intersect / n
            else:
                sIJ_intersect = np.sum((dI > d) & (dJ > d), 1)
                sIJ_overlap = 1 - (sIJ_intersect / n)
            Dmp_list[i] = sIJ_overlap
                 
            Dmp = np.zeros(np.shape(self.D), dtype=self.D.dtype)
            for i in range(n-1):
                j_idx = np.arange(i+1, n)
                Dmp[i, j_idx] = Dmp_list[i]
                Dmp[j_idx, i] = Dmp_list[i]
                
            if self.isSimilarityMatrix:
                for i in range(n):
                    Dmp[i, i] = self.self_value
            
        return Dmp
    
    def _partial_mp_gaussi_sparse(self, batch, matrix, idx, n, mu, sd, verbose):
        Dmp = lil_matrix((len(batch), self.D.shape[1]), dtype=np.float32)
        
        #non-vectorized code
        for i, b in enumerate(batch):
            if verbose and ((batch[i]+1)%1000 == 0 or batch[i]+1==n or i==len(batch)-1 or i==0):
                self.log.message("MP_gaussi_sparse: {} of {}. On {}.".format(
                                batch[i]+1, n, mp.current_process().name, flush=True))  # @UndefinedVariable
            for j in range(b+1, n):
                if matrix[b, j] > 0:       
                    if self.isSimilarityMatrix:
                        p1 = norm.cdf(matrix[b, j], mu[b], sd[b])
                        p2 = norm.cdf(matrix[j, b], mu[j], sd[j])
                        Dmp[i, j] = (p1 * p2).ravel()
                    else:
                        p1 = 1 - norm.cdf(matrix[b, j], mu[b], sd[b])
                        p2 = 1 - norm.cdf(matrix[j, b], mu[j], sd[j])
                        Dmp[i, j] = (1 - p1 * p2).ravel()
                                        
        #=======================================================================
        # #non-vectorized code
        # for i in range(n):
        #     if verbose and ((i+1)%1000 == 0 or i+1==n):
        #         self.log.message("MP_gaussi: {} of {}."
        #                          .format(i+1, n), flush=True)
        #     for j in range(i+1, n):
        #         if self.D[i, j] > 0:       
        #             if self.isSimilarityMatrix:
        #                 p1 = norm.cdf(self.D[i, j], mu[i], sd[i])
        #                 p2 = norm.cdf(self.D[j, i], mu[j], sd[j])
        #                 Dmp[i, j] = (p1 * p2).ravel()
        #             else:
        #                 p1 = 1 - norm.cdf(self.D[i, j], mu[i], sd[i])
        #                 p2 = 1 - norm.cdf(self.D[j, i], mu[j], sd[j])
        #                 Dmp[i, j] = (1 - p1 * p2).ravel()
        #             Dmp[j, i] = Dmp[i, j]
        #=======================================================================
        
        return batch, Dmp
        
        
    def mp_gaussi_sparse(self, train_set_mask, verbose, n_jobs):
        n = self.D.shape[0]
        
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
            self.log.error("MP only supports missing values as zeros. Aborting.", flush=True)
            return
        sd = np.sqrt(va)
        del va
                
        Dmp = lil_matrix(self.D.shape)
        
        # Parallelization
        if n_jobs == -1: # take all cpus
            NUMBER_OF_PROCESSES = mp.cpu_count()  # @UndefinedVariable
        else:
            NUMBER_OF_PROCESSES = n_jobs
        tasks = []
        
        batches = _get_weighted_batches(n, NUMBER_OF_PROCESSES)
        
        for idx, batch in enumerate(batches):
            matrix = self.D#.copy()
            tasks.append((self._partial_mp_gaussi_sparse, (batch, matrix, idx, n, mu, sd, verbose)))   
        
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

        # vectorized code, does not ignore zeros
        #=======================================================================
        # for i in range(n):
        #     if verbose and ((i+1)%1000 == 0 or i+1==n):
        #         self.log.message("MP_gaussi: {} of {}."
        #                          .format(i+1, n), flush=True)
        #     j_idx = np.arange(i+1, n)
        #     #j_len = np.size(j_idx)
        #     
        #     Dij = self.D[i, j_idx].toarray().ravel() #Extract dense rows temporarily
        #     Dji = self.D[j_idx, i].toarray().ravel() #for vectorization below.
        #     
        #     p1 = norm.cdf(Dij, mu[i], sd[i]) # mu, sd broadcasted
        #     p1[Dij==0] = 0
        #     del Dij
        #     p2 = norm.cdf(Dji, 
        #                   mu[j_idx], 
        #                   sd[j_idx])
        #     p2[Dji==0] = 0
        #     del Dji
        #     #del mu, sd # with del mu, sd, error in line with mu broadcasting
        #     tmp = (p1 * p2).ravel()
        #     Dmp[i, i] = self.self_value
        #     Dmp[i, j_idx] = tmp            
        #     Dmp[j_idx, i] = tmp[:, np.newaxis]   
        #     del tmp, j_idx
        #=======================================================================

    def mp_gaussi(self, train_set_mask=None, verbose=False, 
                  sample_size=0, n_jobs=-1):
        """Compute Mutual Proximity modeled with independent Gaussians (fast). 
    
        Set sample_size=SIZE to estimate Gaussian distribution from SIZE 
        samples. Default sample_size=0: use all points."""
        
        if n_jobs is not None:
            self.log.error("MP gaussi does not support parallelization so far.", flush=True)
            
        if self.isSimilarityMatrix:
            self.log.warning("Similarity-based I.Gaussian MP support is still experimental.", flush=True)
        
        n = np.size(self.D, 0)
        if not issparse(self.D):
            np.fill_diagonal(self.D, self.self_value)
        else: 
            # need to set self values later on the fly
            return self.mp_gaussi_sparse(train_set_mask, verbose, n_jobs)
        
        self.log.warning("MP gaussi does not support parallel execution for "
                         "dense matrices atm. Continuing with 1 process.", flush=True)
                
        # Calculate mean and std
        if verbose:
            self.log.message('Calculating distribution parameters.'
                            'Number of samples for parameter estimation: {}'.
                            format(sample_size), flush=True)
        if self.mv is not None:
            self.log.warning("MP gaussi does not support missing value handling"
                             " for dense matrices atm.", flush=True)
        if sample_size != 0:
            samples = np.random.shuffle(train_set_mask)[0:sample_size]
            mu = np.mean(self.D[samples], 0)
            sd = np.std(self.D[samples], 0, ddof=1)
        else:
            mu = np.mean(self.D[train_set_mask], 0)
            sd = np.std(self.D[train_set_mask], 0, ddof=1)             
    
        Dmp = np.zeros_like(self.D)
        for i in range(n):
            if verbose and ((i+1)%1000 == 0 or i+1==n):
                self.log.message("MP_gaussi: {} of {}."
                                 .format(i+1, n), flush=True)
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx)
            
            if self.isSimilarityMatrix:
                p1 = norm.cdf(self.D[i, j_idx], \
                              np.tile(mu[i], (1, j_len)), \
                              np.tile(sd[i], (1, j_len)))
                p2 = norm.cdf(self.D[j_idx, i].T, \
                              mu[j_idx], \
                              sd[j_idx])
                Dmp[i, i] = self.self_value
                Dmp[i, j_idx] = (p1 * p2).ravel()
            else:
                p1 = 1 - norm.cdf(self.D[i, j_idx], \
                                  np.tile(mu[i], (1, j_len)), \
                                  np.tile(sd[i], (1, j_len)))
                p2 = 1 - norm.cdf(self.D[j_idx, i].T, \
                                  mu[j_idx], \
                                  sd[j_idx])
                Dmp[i, j_idx] = (1 - p1 * p2).ravel()
                
            Dmp[j_idx, i] = Dmp[i, j_idx]
    
        return Dmp

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
            #Dij = matrix[b, j_idx].toarray().ravel() #Extract dense rows temporarily
            #Dji = matrix[j_idx, b].toarray().ravel() #for vectorization below.
            
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
            
        #=======================================================================
        # for i in range(NUMBER_OF_PROCESSES):  # @UnusedVariable
        #     if verbose:
        #         self.log.message("Finalizing MP process {}".format(i), flush=True)
        #     task_queue.put('STOP')
        #=======================================================================
         
        for p in processes:
            p.join()
        
        #=======================================================================
        # import os
        # folder = '/tmp/feldbauer/hubPDBs2/'
        # filename = 'D_mpgammai_triu'
        # os.makedirs(folder, exist_ok=True)
        # from scipy import io
        # if verbose:
        #     self.log.message("Saving Dmp upper to {} as {}".format(folder, filename), flush=True)
        # io.mmwrite(folder + filename, Dmp)
        #=======================================================================
        
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
    
    def local_gamcdf(self, x, a, b):
        a[a<0] = np.nan
        b[b<=0] = np.nan
        x[x<0] = 0
        
        # don't calculate gamcdf for missing values
        if self.mv == 0:
            nz = x>0
            z = x[nz] / b[nz]
            p = np.zeros_like(x)
            p[nz] = gammainc(a[nz], z)
        else:
            z = x / b
            p = gammainc(a, z)
        return p
    
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
    from hub_toolbox import MutualProximity as MutProx
    h = Hubness.Hubness(D, 5, True)
    Sn, _, _ = h.calculate_hubness()
    print("Hubness:", Sn)
    
    #===========================================================================
    # mp1 = MutProx.MutualProximity(D, True)
    # Dmp1 = mp1.calculate_mutual_proximity(MutProx.Distribution.gammai, None, True, False, 0, None, empspex=True)
    # h = Hubness.Hubness(Dmp1, 5, True)
    # Sn, _, _ = h.calculate_hubness()
    # print("Hubness (sequential):", Sn)
    #===========================================================================
    mp2 = MutualProximity(D, isSimilarityMatrix=True)
    Dmp2 = mp2.calculate_mutual_proximity(Distribution.empiric, None, True, 0, empspex=False, n_jobs=4)
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
    