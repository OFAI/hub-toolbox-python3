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
import time
from enum import Enum
from hub_toolbox import IO, Logging

class Distribution(Enum):
    empiric = 'empiric'
    gauss = 'gauss'
    gaussi = 'gaussi'
    gammai = 'gammai'


class MutualProximity():
    """Transform a distance matrix with Mutual Proximity.
    
    """
    
    def __init__(self, D):
        self.D = IO.copy_D_or_load_memmap(D, writeable=True)
        self.log = Logging.ConsoleLogging()
        
    def calculate_mutual_proximity(self, distrType=None, test_set_mask=None, 
                                   verbose=False, enforce_disk=False,
                                   sample_size=0, filename=None,
                                   isSimilarityMatrix=False):
        """Apply MP on a distance matrix."""
        
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)
            
        if distrType is None:
            self.log.message("No Mutual Proximity type given. "
                             "Using: Distribution.empiric \n"
                             "For fast results use: Distribution.gaussi")
            Dmp = self.mp_empiric(train_set_mask, verbose, isSimilarityMatrix)
        else:
            if distrType == Distribution.empiric:
                Dmp = self.mp_empiric(train_set_mask, verbose, isSimilarityMatrix)
            elif distrType == Distribution.gauss:
                Dmp = self.mp_gauss(train_set_mask, verbose, isSimilarityMatrix)
            elif distrType == Distribution.gaussi:
                Dmp = self.mp_gaussi(train_set_mask, verbose, enforce_disk, 
                                     sample_size, filename, isSimilarityMatrix)
            elif distrType == Distribution.gammai:
                Dmp = self.mp_gammai(train_set_mask, verbose, isSimilarityMatrix)
            else:
                self.log.warning("Valid Mutual Proximity type missing!\n"+\
                             "Use: \n"+\
                             "mp = MutualProximity(D, Distribution.empiric|"+\
                             "Distribution.gauss|Distribution.gaussi|"+\
                             "Distribution.gammi)\n"+\
                             "Dmp = mp.calculate_mutual_proximity()")
                Dmp = np.array([])
       
        return Dmp
         
    def mp_empiric(self, train_set_mask=None, verbose=False, 
                   isSimilarityMatrix=False):
        """Compute Mutual Proximity distances with empirical data (slow)."""   
        #TODO implement train_set_mask!
        if not np.all(train_set_mask):
            print("WARNING: MP empiric does not support train/test splits yet.")
            return
        #TODO implement similarity based MP
        if isSimilarityMatrix:
            self.log.warning("Similarity-based empiric MP support is still experimental.")
            
        if isinstance(self.D, np.memmap): # work on disk
            from tempfile import mkstemp
            filename = mkstemp(suffix='pytmp')[1] # [0]... fd, [1]... filename
            self.log.message("Writing rescaled distance matrix to file:", filename)
            Dmp = np.memmap(filename, dtype='float64', mode='w+', shape=self.D.shape)
            n = self.D.shape[0]
            np.fill_diagonal(self.D, 0)
            
            tic = time.clock() 
            for i in range(n-1):
                if verbose and ((i+1)%1000 == 0 or i==n):
                    toc = time.clock() - tic
                    self.log.message("MP_empiric: {} of {}. Took {:.3} "
                                    "seconds.".format(i+1, n-1, toc))
                    tic = time.clock()
                j_idx = np.arange(i+1, n)
                j_len = np.size(j_idx, 0)
               
                dI = np.tile(self.D[i, :], (j_len, 1))
                dJ = self.D[j_idx, :]
                d = np.tile(self.D[j_idx, i][:, np.newaxis], (1, n))
                
                if isSimilarityMatrix:
                    sIJ_intersect = np.sum((dI < d) & (dJ < d), 1)
                    sIJ_overlap = sIJ_intersect / n
                else: 
                    sIJ_intersect = np.sum((dI > d) & (dJ > d), 1)
                    sIJ_overlap = 1 - (sIJ_intersect / n)
                Dmp[i, j_idx] = sIJ_overlap 
            Dmp += Dmp.T   
              
        else: # work in memory
            np.fill_diagonal(self.D, 0)
            n = np.shape(self.D)[0]
            Dmp_list = [0 for i in range(n)]
             
            tic = time.clock()
            for i in range(n-1):
                if verbose and ((i+1)%1000 == 0 or i==n):
                    toc = time.clock() - tic
                    self.log.message("MP_empiric: {} of {}. Took {:.3} "
                                    "seconds.".format(i+1, n-1, toc))
                    tic = time.clock() 
                # Select only finite distances for MP
                j_idx = np.arange(i+1, n)
                j_len = np.size(j_idx, 0)
                 
                dI = np.tile(self.D[i, :], (j_len, 1))
                dJ = self.D[j_idx, :]
                d = np.tile(self.D[j_idx, i][:, np.newaxis], (1, n))
                 
                if isSimilarityMatrix:
                    sIJ_intersect = np.sum((dI < d) & (dJ < d), 1)
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
            
        return Dmp
    
    def mp_gauss(self, train_set_mask=None, verbose=False,
                 isSimilarityMatrix=False):
        """Compute Mutual Proximity distances with Gaussian model (very slow)."""
        #TODO implement similarity based MP
        if isSimilarityMatrix:
            raise ValueError("Gaussian MP does not yet support similarity matrices.")
        
        np.fill_diagonal(self.D, 0)
        mu = np.mean(self.D[train_set_mask], 0)
        sd = np.std(self.D[train_set_mask], 0, ddof=1)
                
        #Code for the BadMatrixSigma error
        eps = np.spacing(1)
        epsmat = np.array([[1e5 * eps, 0], [0, 1e5 * eps]])
                
        if isinstance(self.D, np.memmap):
            from tempfile import mkstemp
            filename = mkstemp(suffix='pytmp')[1] # [0]... fd, [1]... filename
            self.log.message("Writing rescaled distance matrix to file:", filename)
            Dmp = np.memmap(filename, dtype='float64', mode='w+', shape=self.D.shape)
        else:
            Dmp = np.zeros(np.shape(self.D), dtype=self.D.dtype)
        n = np.size(self.D, 0)
        
        tic = time.clock()
        for i in range(n):
            if verbose and ((i+1)%1000 == 0 or i+1==n):
                toc = time.clock() - tic
                self.log.message("MP_gauss: {} of {}. Took {:.3} "
                                "seconds.".format(i+1, n, toc))
                tic = time.clock()
            for j in range(i+1, n):
                c = np.cov(self.D[[i,j], :])
                x = np.array([self.D[i, j], self.D[j, i]])
                m = np.array([mu[i], mu[j]])
                
                p1 = norm.cdf(self.D[j, i], mu[i], sd[i])
                p2 = norm.cdf(self.D[j, i], mu[j], sd[j])
                
                low = np.tile(np.finfo(np.float32).min, 2)
                p12 = mvn.mvnun(low, x, m, c)[0] # [0]...p, [1]...inform
                if np.isnan(p12):
                    c += epsmat*1e7 
                    p12 = mvn.mvnun(low, x, m, c)[0]
                assert not np.isnan(p12), "p12 is NaN: i={}, j={}".format(i, j)
                Dmp[j, i] = p1 + p2 - p12
                Dmp[i, j] = Dmp[j, i]

        return Dmp
    
    def mp_gaussi(self, train_set_mask=None, verbose=False, enforce_disk=False,
                  sample_size=0, filename=None, isSimilarityMatrix=False):
        """Compute Mutual Proximity modeled with independent Gaussians (fast). 
        Use enforce_disk=True to use memory maps for matrices that do not fit 
        into main memory.
        Set sample_size=SIZE to estimate Gaussian distribution from SIZE 
        samples. Default sample_size=0: use all points."""
        #TODO implement similarity based MP
        if isSimilarityMatrix:
            raise ValueError("I.Gaussian MP does not yet support similarity matrices.")
        
        if verbose:
            self.log.message('Mutual proximity rescaling started.', flush=True)
        n = np.size(self.D, 0)
        if not isinstance(self.D, np.memmap):
            np.fill_diagonal(self.D, 0)
        # else: do this later on the fly
        
        # Get memory info
        free_mem = IO.FreeMemLinux(unit='k').user_free
        req_mem = self.D.nbytes
        mem_ratio = int(req_mem / free_mem ) 
                
        # Calculate mean and std
        if mem_ratio < 2.0 and not enforce_disk:
            if verbose:
                self.log.message('Calculating distribution parameters in memory.'
                                'Number of samples for parameter estimation: {}'.
                                format(sample_size))
            if sample_size != 0:
                samples = np.random.shuffle(train_set_mask)[0:sample_size]
                mu = np.mean(self.D[samples], 0)
                sd = np.std(self.D[samples], 0, ddof=1)
            else:
                mu = np.mean(self.D[train_set_mask], 0)
                sd = np.std(self.D[train_set_mask], 0, ddof=1)
        # ... using tertiary memory, if necessary
        else:
            if verbose:
                self.log.message('Calculating Gaussian parameters on disk. '
                                'Number of samples for parameter estimation: '
                                '{}'.format(sample_size))
            if not np.all(train_set_mask) and isinstance(self.D, np.memmap):
                raise NotImplementedError("Using train/test splits and working "
                                          "and working on disk at the same time"
                                          "not supported atm.")
            mu = np.zeros(self.D.shape[1])
            sd = np.zeros_like(mu)
            
            if sample_size != 0:
                idx = np.arange(self.D.shape[0])
                np.random.shuffle(idx)
                samples = idx[0:sample_size]
                
                tic = time.clock()
                for i, row in enumerate(self.D[samples].T):
                    if verbose and ((i+1)%10000 == 0 or i+1 == n):
                        toc = time.clock() - tic
                        self.log.message("MP_gaussi mean/std: "
                                        "{} of {}. Took {:.3} seconds."
                                        .format(i+1, n, toc), flush=True)
                        tic = time.clock()
                    mu[i] = row.mean()
                    sd[i] = row.std(ddof=1)    
            else:
                tic = time.clock()
                for i, row in enumerate(self.D.T):
                    if verbose and ((i+1)%10000 == 0 or i+1 == n):
                        toc = time.clock() - tic
                        self.log.message("MP_gaussi mean/std: "
                                        "{} of {}. Took {:.3} seconds."
                                        .format(i+1, n, toc), flush=True)
                        tic = time.clock()
                    mu[i] = row.mean()
                    sd[i] = row.std(ddof=1)                            
        
        # work on disk
        if isinstance(self.D, np.memmap) or enforce_disk:
            if filename is None:
                # create distance matrix on disk
                from tempfile import mkstemp
                filename = mkstemp(suffix='pytmp')[1] # [0]... fd, [1]... filename
            #else: 
            #    filename was provided via function call
            self.log.message("Writing rescaled distance matrix to file:", filename)
            Dmp = np.memmap(filename, dtype='float64', mode='w+', shape=self.D.shape)
            # determine number of rows that fit into free memory
            #------------- max_rows = int(free_mem / n / 8) # 8 byte per float64
            # take_rows = int(max_rows / 8) # allow the current matrices to occupy 1/4 of the available memory
            #-------------------------- nr_batches = int(np.ceil(n / take_rows))
            nr_batches, take_rows = IO.matrix_split(n, n, 8, 4)
            if verbose:
                self.log.message('Divided {}x{} matrix into {} '
                                'batches of {} rows each.'.
                                format(n, n, nr_batches, take_rows), flush=True)
            tic = time.clock()
            i = 0
            # work on submatrices, trying to minimize disk I/O
            for h in range(nr_batches):
                idx_start = h*take_rows
                idx_stop = (h+1)*take_rows
                if idx_stop > n:
                    idx_stop = n
                idx = np.arange(idx_start, idx_stop).astype(np.int)
                current_rows = np.copy(self.D[idx, :])
                current_cols = np.copy(self.D[:, idx])
                # fill diag with 0 (self distance = 0)
                for r, c in enumerate(idx):
                    current_rows[r, c] = 0
                for c, r in enumerate(idx):
                    current_cols[r, c] = 0 
                updated_rows = np.zeros_like(current_rows)
                # calculations on submatrix in memory
                for inner_row in range(len(idx)):
                    if verbose and ((i+1)%1000 == 0 or i+1==n):
                        toc = time.clock() - tic
                        self.log.message("MP_gaussi: {} of {}. Took {:.3} "
                                        "seconds.".format(i+1, n, toc), 
                                        flush=True)
                        tic = time.clock()
                    j_idx = np.arange(i+1, n)
                    j_len = np.size(j_idx)
                    
                    p1 = 1 - norm.cdf(current_rows[inner_row, j_idx], \
                                      np.tile(mu[i], (1, j_len)), \
                                      np.tile(sd[i], (1, j_len)))        
                    p2 = 1 - norm.cdf(current_cols[j_idx, inner_row].T, \
                                      mu[j_idx], sd[j_idx])
                    updated_rows[inner_row, j_idx] = (1 - p1 * p2).ravel()
                    i += 1
                # writing changes for many rows at once
                Dmp[idx, :] = updated_rows
            # make symmetric distance matrix
            #Dmp += Dmp.T # inefficient
            # hopefully faster using submatrices:
            for h in range(nr_batches):
                if verbose:
                    self.log.message('MP finalization: batch {}/{}.'.
                                     format(h, nr_batches-1), flush=True)
                row_start = h*take_rows
                row_stop = (h+1)*take_rows
                if row_stop > n:
                    row_stop = n
                row_idx = np.arange(row_start, row_stop).astype(np.int)
                col_start = row_start
                col_stop = n
                col_idx = np.arange(col_start, col_stop).astype(np.int)
                
                current_rows = np.copy(Dmp[np.ix_(row_idx, col_idx)])
                current_cols = np.copy(Dmp[np.ix_(col_idx, row_idx)])
                # mirroring the matrix
                current_cols += current_rows.T
                Dmp[np.ix_(col_idx, row_idx)] = current_cols 
    
        else: # work in memory
            Dmp = np.zeros_like(self.D)
            tic = time.clock()
            for i in range(n):
                if verbose and ((i+1)%1000 == 0 or i+1==n):
                    toc = time.clock() - tic
                    self.log.message("MP_gaussi: {} of {}. Took {:.3} "
                                    "seconds.".format(i+1, n, toc), flush=True)
                    tic = time.clock()
                j_idx = np.arange(i+1, n)
                j_len = np.size(j_idx)
                
                p1 = 1 - norm.cdf(self.D[i, j_idx], \
                                  np.tile(mu[i], (1, j_len)), \
                                  np.tile(sd[i], (1, j_len)))
                p2 = 1 - norm.cdf(self.D[j_idx, i].T, \
                                  mu[j_idx], sd[j_idx])
                Dmp[i, j_idx] = (1 - p1 * p2).ravel()
                Dmp[j_idx, i] = Dmp[i, j_idx]
    
        return Dmp
    
    def mp_gammai(self, train_set_mask=None, verbose=False,
                  isSimilarityMatrix=False):
        """Compute Mutual Proximity modeled with independent Gamma distributions."""
        #TODO implement similarity based MP
        if isSimilarityMatrix:
            raise ValueError("I.Gamma MP does not yet support similarity matrices.")
        
        np.fill_diagonal(self.D, 0)
        mu = np.mean(self.D[train_set_mask], 0)
        va = np.var(self.D[train_set_mask], 0, ddof=1)
        A = (mu**2) / va
        B = va / mu
        
        if isinstance(self.D, np.memmap):
            from tempfile import mkstemp
            filename = mkstemp(suffix='pytmp')[1] # [0]... fd, [1]... filename
            self.log.message("Writing rescaled distance matrix to file:", filename)
            Dmp = np.memmap(filename, dtype='float64', mode='w+', shape=self.D.shape)
        else:
            Dmp = np.zeros_like(self.D)
        n = np.size(self.D, 0)
        
        tic = time.clock()
        for i in range(n):
            if verbose and ((i+1)%1000 == 0 or i+1==n):
                toc = time.clock() - tic
                self.log.message("MP_gammai: {} of {}. Took {:.3} "
                                "seconds.".format(i+1, n, toc))
                tic = time.clock()
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx)
            
            p1 = 1 - self.local_gamcdf(self.D[i, j_idx], \
                                       np.tile(A[i], (1, j_len)), \
                                       np.tile(B[i], (1, j_len)))
            p2 = 1 - self.local_gamcdf(self.D[j_idx, i].T, A[j_idx], B[j_idx])
            
            Dmp[i, j_idx] = (1 - p1 * p2).ravel()
            Dmp[j_idx, i] = Dmp[i, j_idx]               
        
        return Dmp
    
    def local_gamcdf(self, x, a, b):
        a[a<0] = np.nan
        b[b<=0] = np.nan
        x[x<0] = 0
        z = x / b
        p = gammainc(a, z)
        return p
    