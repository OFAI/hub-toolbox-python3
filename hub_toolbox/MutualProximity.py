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
from scipy.stats import norm, mvn
from scipy.sparse import dok_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.base import issparse
from hub_toolbox import IO, Logging
import sys
# DEPRECATED
from enum import Enum
from scipy.sparse.lil import lil_matrix

def mutual_proximity_empiric(D:np.ndarray, metric:str='distance', 
                             test_set_ind:np.ndarray=None, verbose:int=0):
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
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")  
    if metric == 'similarity':
        self_value = 1
    else:
        self_value = 0
        if issparse(D):
            raise ValueError("MP sparse only supports similarity matrices.")
    if test_set_ind is None:
        pass # TODO implement
        #train_set_ind = slice(0, n)
    else:
        raise NotImplementedError("MP empiric does not yet support train/"
                                  "test splits.")
        #train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    D = D.copy()
    
    if issparse(D):
        log.warning("Please use MutualProximity_parallel for sparse MP.")
        return _mutual_proximity_empiric_sparse(D, test_set_ind, verbose, log)
        
    # ensure correct self distances (NOT done for sparse matrices!)
    np.fill_diagonal(D, self_value)
    
    D_mp = np.zeros_like(D)
     
    for i in range(n-1):
        if verbose and ((i+1)%1000 == 0 or i==n-2):
            log.message("MP_empiric: {} of {}.".format(i+1, n-1), flush=True)
        # Select only finite distances for MP
        j_idx = i + 1
         
        dI = D[i, :][np.newaxis, :]
        dJ = D[j_idx:n, :]
        d = D[j_idx:n, i][:, np.newaxis]
         
        if metric == 'similarity':
            D_mp[i, j_idx:] = np.sum((dI <= d) & (dJ <= d), 1) / n
        else: # metric == 'distance':
            D_mp[i, j_idx:] = 1 - (np.sum((dI > d) & (dJ > d), 1) / n)
         
    D_mp += D_mp.T
    np.fill_diagonal(D_mp, self_value)

    return D_mp


def _mutual_proximity_empiric_sparse(S:csr_matrix, 
                                     test_set_ind:np.ndarray=None, 
                                     verbose:int=0,
                                     log=None):
    """MP empiric for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_empiric()
    """
    self_value = 1. # similarity matrix
    n = S.shape[0]        
    nnz = S.nnz
    S_mp = lil_matrix(S.shape)
    
    for i in range(n-1):
        if verbose and ((i+1)%1000 == 0 or i==n-2):
            log.message("MP_empiric: {} of {}.".format(i+1, n-1), flush=True)
        for j in range(i+1, n):
            d = S[j, i]
            if d>0: 
                dI = S[i, :].todense()
                dJ = S[j, :].todense()
                # non-zeros elements
                nz = (dI > 0) & (dJ > 0)  # @UnusedVariable
                #TODO continue...
                sIJ_intersect = ((dI <= d) & (dJ <= d)).sum()
                sIJ_overlap = sIJ_intersect / nnz
                
                S_mp[i, j] = sIJ_overlap
                S_mp[j, i] = sIJ_overlap
            else:
                pass # skip zero entries
    
    for i in range(n):
        S_mp[i, i] = self_value #need to set self values
    
    return S_mp.tocsr()

def mutual_proximity_gauss(D:np.ndarray, metric:str='distance', 
                           test_set_ind:np.ndarray=None, verbose:int=0):
    """Transform a distance matrix with Mutual Proximity (normal distribution).
    
    Applies Mutual Proximity (MP) [1] on a distance/similarity matrix. Gauss 
    variant assumes dependent normal distributions (VERY SLOW).
    The resulting second. distance/similarity matrix should show lower hubness.
    
    Parameters:
    -----------
    D : ndarray
        - ndarray: The n x n symmetric distance or similarity matrix.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix.
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).
        
    Returns:
    --------
    D_mp : ndarray
        Secondary distance MP gauss matrix.
    
    See also:
    ---------
    [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
    Local and global scaling reduce hubs in space. The Journal of Machine 
    Learning Research, 13(1), 2871–2902.
    """
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")  
    if metric == 'similarity':
        log.warning("MP Gauss is untested for similarity matrices. "
                    "Use with caution!")
        self_value = 1
    else: # metric == 'distance':
        self_value = 0  
    if issparse(D):
        log.error("Sparse matrices not supported by MP Gauss.")
        raise TypeError("Sparse matrices not supported by MP Gauss.")
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
    D = D.copy()
    
    np.fill_diagonal(D, self_value)
    
    mu = np.mean(D[train_set_ind], 0)
    sd = np.std(D[train_set_ind], 0, ddof=1)
            
    #Code for the BadMatrixSigma error [derived from matlab]
    eps = np.spacing(1)
    epsmat = np.array([[1e5 * eps, 0], [0, 1e5 * eps]])
            
    D_mp = np.zeros_like(D)
    
    for i in range(n):
        if verbose and ((i+1)%1000 == 0 or i+1==n):
            log.message("MP_gauss: {} of {}.".format(i+1, n))
        for j in range(i+1, n):
            c = np.cov(D[[i,j], :])
            x = np.array([D[i, j], D[j, i]])
            m = np.array([mu[i], mu[j]])
            
            low = np.tile(np.finfo(np.float32).min, 2)
            p12 = mvn.mvnun(low, x, m, c)[0] # [0]...p, [1]...inform
            if np.isnan(p12):
                power = 7
                while np.isnan(p12):
                    c += epsmat * (10**power) 
                    p12 = mvn.mvnun(low, x, m, c)[0]
                    power += 1
                log.warning("p12 is NaN: i={}, j={}. Increased cov matrix by "
                            "O({}).".format(i, j, epsmat[0,0]*(10**power)))
            
            if metric == 'similarity':
                D_mp[j, i] = p12
            else: # distance
                p1 = norm.cdf(D[j, i], mu[i], sd[i])
                p2 = norm.cdf(D[j, i], mu[j], sd[j])
                D_mp[j, i] = p1 + p2 - p12
            D_mp[i, j] = D_mp[j, i]
        if metric == 'similarity':
            D_mp[i, i] = self_value
    
    return D_mp

def mutual_proximity_gaussi(D:np.ndarray, metric:str='distance', 
                            sample_size:int=0, test_set_ind:np.ndarray=None, 
                            verbose:int=0):
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
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")  
    if metric == 'similarity':
        self_value = 1
    else: # metric == 'distance':
        self_value = 0  
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
    D = D.copy()
    
    if verbose:
        log.message('Mutual Proximity Gaussi rescaling started.', flush=True)

    if issparse(D):
        return _mutual_proximity_gaussi_sparse(D, sample_size, test_set_ind, 
                                               verbose, log)

    np.fill_diagonal(D, self_value)
        
    # Calculate mean and std
    if sample_size != 0:
        samples = np.random.shuffle(train_set_ind)[0:sample_size]
        mu = np.mean(D[samples], 0)
        sd = np.std(D[samples], 0, ddof=1)
    else:
        mu = np.mean(D[train_set_ind], 0)
        sd = np.std(D[train_set_ind], 0, ddof=1)
    
    D_mp = np.zeros_like(D)
    for i in range(n):
        if verbose and ((i+1)%1000 == 0 or i+1==n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        j_idx = np.arange(i+1, n)
        j_len = np.size(j_idx)
        
        if metric == 'similarity':
            # TODO change np.tile to broadcasting
            p1 = norm.cdf(D[i, j_idx], \
                          np.tile(mu[i], (1, j_len)), \
                          np.tile(sd[i], (1, j_len)))
            p2 = norm.cdf(self.D[j_idx, i].T, \
                          mu[j_idx], \
                          sd[j_idx])
            D_mp[i, i] = self_value
            D_mp[i, j_idx] = (p1 * p2).ravel()
        else:
            p1 = 1 - norm.cdf(D[i, j_idx], \
                              np.tile(mu[i], (1, j_len)), \
                              np.tile(sd[i], (1, j_len)))
            p2 = 1 - norm.cdf(self.D[j_idx, i].T, \
                              mu[j_idx], \
                              sd[j_idx])
            D_mp[i, j_idx] = (1 - p1 * p2).ravel()
            
        D_mp[j_idx, i] = D_mp[i, j_idx]

    return D_mp

def _mutual_proximity_gaussi_sparse(D:np.ndarray, sample_size:int=0, 
                                    test_set_ind:np.ndarray=None, 
                                    verbose:int=0, log=None):
    """MP gaussi for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gaussi()
    """
    n = D.shape[0]
    self_value = 1 # similarity matrix
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
    from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  # @UnresolvedImport
    mu, var = csr_mean_variance_axis0(D[train_set_ind])
    sd = np.sqrt(var)
    del var
    
    D_mp = dok_matrix(D.shape)

    for i in range(n):
        if verbose and log and ((i+1)%1000 == 0 or i+1==n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        j_idx = np.arange(i+1, n)
        #j_len = np.size(j_idx)
        
        Dij = self.D[i, j_idx].toarray().ravel() #Extract dense rows temporarily
        Dji = self.D[j_idx, i].toarray().ravel() #for vectorization below.
        
        p1 = norm.cdf(Dij, mu[i], sd[i]) # mu, sd broadcasted
        p1[Dij==0] = 0
        del Dij
        p2 = norm.cdf(Dji, mu[j_idx], sd[j_idx])
        p2[Dji==0] = 0
        del Dji
        #del mu, sd # with del mu, sd, error in line with mu broadcasting
        tmp = (p1 * p2).ravel()
        D_mp[i, i] = self_value
        D_mp[i, j_idx] = tmp            
        D_mp[j_idx, i] = tmp[:, np.newaxis]   
        del tmp, j_idx
    
    return D_mp.tocsr()

def mutual_proximity_gammai():
    pass

def _local_gamcdf(self, x, a, b):
    """Gamma CDF"""
    a[a<0] = np.nan
    b[b<=0] = np.nan
    x[x<0] = 0
    z = x / b
    p = gammainc(a, z)
    return p

##############################################################################
#
# DEPRECATED classes
#
class Distribution(Enum):
    """DEPRECATED"""
    empiric = 'empiric'
    gauss = 'gauss'
    gaussi = 'gaussi'
    gammai = 'gammai'

class MutualProximity():
    """DEPRECATED"""
    
    def __init__(self, D, isSimilarityMatrix=False):
        """DEPRECATED"""
        print("DEPRECATED: Please use the appropriate MutualProximity."
              "mutual_proximity_DISTRIBUTIONTYPE() function instead.", 
              file=sys.stderr)
        self.D = IO.copy_D_or_load_memmap(D, writeable=True)
        self.log = Logging.ConsoleLogging()
        if isSimilarityMatrix:
            self.self_value = 1
        else:
            self.self_value = 0
        self.isSimilarityMatrix = isSimilarityMatrix
        
    def calculate_mutual_proximity(self, distrType=None, test_set_mask=None, 
                                   verbose=False, enforce_disk=False,
                                   sample_size=0, filename=None):
        """DEPRECATED"""
        
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)
            
        if distrType is None:
            self.log.message("No Mutual Proximity type given. "
                             "Using: Distribution.empiric. "
                             "For fast results use: Distribution.gaussi")
            Dmp = self.mp_empiric(train_set_mask, verbose)
        else:
            if distrType == Distribution.empiric:
                Dmp = self.mp_empiric(train_set_mask, verbose)
            elif distrType == Distribution.gauss:
                Dmp = self.mp_gauss(train_set_mask, verbose)
            elif distrType == Distribution.gaussi:
                Dmp = self.mp_gaussi(train_set_mask, verbose, enforce_disk, 
                                     sample_size, filename)
            elif distrType == Distribution.gammai:
                Dmp = self.mp_gammai(train_set_mask, verbose)
            else:
                self.log.warning("Valid Mutual Proximity type missing!\n"+\
                             "Use: \n"+\
                             "mp = MutualProximity(D, Distribution.empiric|"+\
                             "Distribution.gauss|Distribution.gaussi|"+\
                             "Distribution.gammi)\n"+\
                             "Dmp = mp.calculate_mutual_proximity()")
                Dmp = np.array([])
       
        return Dmp
         

    def mp_empiric_sparse(self, train_set_mask=None, verbose=False):
        """DEPRECATED"""
        return mutual_proximity_empiric(self.D, 'similarity', None, verbose)
    
    def mp_empiric(self, train_set_mask=None, verbose=False):
        """DEPRECATED"""  
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance' 
        if train_set_mask is not None:
            test_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        return mutual_proximity_empiric(self.D, metric, test_set_mask, verbose)
    
    def mp_gauss(self, train_set_mask=None, verbose=False):
        """DEPRECATED"""
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance' 
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        else:#
            test_set_ind = None
        return mutual_proximity_gauss(self.D, metric, test_set_ind, verbose)
        
        
    def mp_gaussi_sparse(self, train_set_mask, verbose):
        """DEPRECATED"""
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        else:
            test_set_ind = None
        return mutual_proximity_gaussi(self.D, 0, test_set_ind, verbose)

    def mp_gaussi(self, train_set_mask=None, verbose=False, enforce_disk=False,
                  sample_size=0, filename=None):
        """DEPRECATED"""
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance' 
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        else:#
            test_set_ind = None
        return mutual_proximity_gaussi(self.D, metric, sample_size, test_set_ind, verbose)
    

    def mp_gammai_sparse(self, train_set_mask, verbose):
        # mean, variance WITH zero values
        #=======================================================================
        # from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  
        # mu, va = csr_mean_variance_axis0(self.D[train_set_mask])
        #=======================================================================
        
        # mean, variance WITHOUT zero values (missing values)
        # TODO implement train_test split
        mu = np.array(self.D.sum(0) / self.D.getnnz(0)).ravel()
        X = self.D.copy()
        X.data **= 2
        E1 = np.array(X.sum(0) / X.getnnz(0)).ravel()
        del X
        va = E1 - mu**2
        del E1
        
        A = (mu**2) / va
        B = va / mu
        del mu, va
        A[A<0] = np.nan
        B[B<=0] = np.nan

        Dmp = dok_matrix(self.D.shape, dtype=np.float32)
        n = self.D.shape[0]
        
        for i in range(n):
            if verbose and ((i+1)%1000 == 0 or i+1==n):
                self.log.message("MP_gammai: {} of {}".format(i+1, n), flush=True)
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx)
             

            Dij = self.D[i, j_idx].toarray().ravel() #Extract dense rows temporarily
            Dji = self.D[j_idx, i].toarray().ravel() #for vectorization below.
            
            p1 = self.local_gamcdf(Dij, \
                                   np.tile(A[i], (1, j_len)), \
                                   np.tile(B[i], (1, j_len)))
            del Dij
            p2 = self.local_gamcdf(Dji, 
                                   A[j_idx], 
                                   B[j_idx])
            del Dji#, A, B
            tmp = (p1 * p2).ravel()
            Dmp[i, i] = self.self_value
            Dmp[i, j_idx] = tmp     
            Dmp[j_idx, i] = tmp[:, np.newaxis]
            del tmp, j_idx
               
        return Dmp.tocsr()
    
    
    def mp_gammai(self, train_set_mask=None, verbose=False):
        """Compute Mutual Proximity modeled with independent Gamma distributions."""
        if self.isSimilarityMatrix:
            self.log.warning("Similarity-based I.Gamma MP support is still experimental.")
        if verbose:
            self.log.message('Mutual proximity rescaling started.', flush=True)
        
        if not issparse(self.D):
            np.fill_diagonal(self.D, self.self_value)
        else:
            #self.log.error("Sparse matrices not supported yet.")
            return self.mp_gammai_sparse(train_set_mask, verbose)
        
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
    
