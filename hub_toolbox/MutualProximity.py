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

import sys
# DEPRECATED
from enum import Enum
import numpy as np
from scipy.special import gammainc  # @UnresolvedImport
from scipy.stats import norm, mvn
from scipy.sparse import lil_matrix, csr_matrix, issparse, triu
from hub_toolbox import IO, Logging

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
          NOTE: In case of sparse D, zeros are interpreted as missing values 
                and ignored during calculations. Thus, results may differ 
                from using a dense version.
    
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
    # Initialization
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    
    # Check input
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
    elif not np.all(~test_set_ind):
        raise NotImplementedError("MP empiric does not yet support train/"
                                  "test splits.")
        #train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

    # Start MP
    D = D.copy()
    
    if issparse(D):
        return _mutual_proximity_empiric_sparse(D, test_set_ind, verbose, log)
        
    # ensure correct self distances (NOT done for sparse matrices!)
    np.fill_diagonal(D, self_value)
    
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
            D_mp[i, j_idx:] = np.sum((dI <= d) & (dJ <= d), 1) / n
        else: # metric == 'distance':
            D_mp[i, j_idx:] = 1 - (np.sum((dI > d) & (dJ > d), 1) / n)
         
    # Mirror, so that matrix is symmetric
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
    S_mp = lil_matrix(S.shape)
    
    for i, j in zip(*triu(S).nonzero()):
        if verbose and log and ((i+1)%1000 == 0 or i == n-2):
            log.message("MP_empiric: {} of {}.".format(i+1, n-1), flush=True)
        d = S[j, i]
        dI = S.getrow(i).toarray()
        dJ = S.getrow(j).toarray()
        nz = (dI > 0) & (dJ > 0)
        S_mp[i, j] = (nz & (dI <= d) & (dJ <= d)).sum() / nz.sum()
    
    S_mp += S_mp.T
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
    # Initialization
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    
    # Checking input
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")  
    if metric == 'similarity':
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
        
    # Start MP
    D = D.copy()
    
    np.fill_diagonal(D, self_value)
    
    mu = np.mean(D[train_set_ind], 0)
    sd = np.std(D[train_set_ind], 0, ddof=1)
            
    #Code for the BadMatrixSigma error [derived from matlab]
    eps = np.spacing(1)
    epsmat = np.array([[1e5 * eps, 0], [0, 1e5 * eps]])
            
    D_mp = np.zeros_like(D)
    
    # MP Gauss
    for i in range(n):
        if verbose and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gauss: {} of {}.".format(i+1, n))
        for j in range(i+1, n):
            c = np.cov(D[[i, j], :])
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
                            "O({}).".format(i, j, epsmat[0, 0]*(10**power)))
            
            if metric == 'similarity':
                D_mp[i, j] = p12
            else: # distance
                p1 = norm.cdf(D[i, j], mu[i], sd[i])
                p2 = norm.cdf(D[i, j], mu[j], sd[j])
                D_mp[i, j] = p1 + p2 - p12
    D_mp += D_mp.T
    np.fill_diagonal(D_mp, self_value)
    
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
          NOTE: In case of sparse D, zeros are interpreted as missing values 
                and ignored during calculations. Thus, results may differ 
                from using a dense version.
    
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
    # Initialization   
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    
    # Checking input
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
    
    # Start MP Gaussi    
    if verbose:
        log.message('Mutual Proximity Gaussi rescaling started.', flush=True)
    D = D.copy()

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
    
    # MP Gaussi
    D_mp = np.zeros_like(D)
    for i in range(n):
        if verbose and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)
        
        if metric == 'similarity':
            p1 = norm.cdf(D[i, j_idx], mu[i], sd[i])
            p2 = norm.cdf(D[j_idx, i], mu[j_idx], sd[j_idx])
            D_mp[i, j_idx] = (p1 * p2).ravel()
        else:
            p1 = 1 - norm.cdf(D[i, j_idx], mu[i], sd[i])
            p2 = 1 - norm.cdf(D[j_idx, i], mu[j_idx], sd[j_idx])
            D_mp[i, j_idx] = (1 - p1 * p2).ravel()
    D_mp += D_mp.T        
    np.fill_diagonal(D_mp, self_value)
    
    return D_mp

def _mutual_proximity_gaussi_sparse(S:np.ndarray, sample_size:int=0, 
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
    from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  # @UnresolvedImport
    mu, var = csr_mean_variance_axis0(S[train_set_ind])
    sd = np.sqrt(var)
    del var
    
    S_mp = lil_matrix(S.shape)

    for i in range(n):
        if verbose and log and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gaussi: {} of {}.".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)
        
        S_ij = S[i, j_idx].toarray().ravel() #Extract dense rows temporarily
        S_ji = S[j_idx, i].toarray().ravel() #for vectorization below.
        
        p1 = norm.cdf(S_ij, mu[i], sd[i]) # mu, sd broadcasted
        p1[S_ij == 0] = 0
        del S_ij
        p2 = norm.cdf(S_ji, mu[j_idx], sd[j_idx])
        p2[S_ji == 0] = 0
        del S_ji
        tmp = np.empty(n-i)
        tmp[0] = self_value / 2. 
        tmp[1:] = (p1 * p2).ravel()
        S_mp[i, i:] = tmp            
        del tmp, j_idx
    
    S_mp += S_mp.T
    return S_mp.tocsr()

def mutual_proximity_gammai(D:np.ndarray, metric:str='distance', 
                            test_set_ind:np.ndarray=None, verbose:int=0):
    """Transform a distance matrix with Mutual Proximity (indep. Gamma distr.).
    
    Applies Mutual Proximity (MP) [1] on a distance/similarity matrix. Gammai 
    variant assumes independent Gamma distributed distances (FAST).
    The resulting second. distance/similarity matrix should show lower hubness.
    
    Parameters:
    -----------
    D : ndarray or csr_matrix
        - ndarray: The n x n symmetric distance or similarity matrix.
        - csr_matrix: The n x n symmetric similarity matrix.
          NOTE: In case of sparse D, zeros are interpreted as missing values 
                and ignored during calculations. Thus, results may differ 
                from using a dense version.
    
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
        Secondary distance MP gammai matrix.
    
    See also:
    ---------
    [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
    Local and global scaling reduce hubs in space. The Journal of Machine 
    Learning Research, 13(1), 2871–2902.
    """   
    # Initialization
    n = D.shape[0]
    log = Logging.ConsoleLogging()
    
    # Checking input
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
    
    # Start MP 
    if verbose:
        log.message('Mutual proximity Gammai rescaling started.', flush=True)
    D = D.copy()
    
    if issparse(D):
        return _mutual_proximity_gammai_sparse(D, test_set_ind, verbose, log)

    np.fill_diagonal(D, self_value)
    
    mu = np.mean(D[train_set_ind], 0)
    va = np.var(D[train_set_ind], 0, ddof=1)
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

def _mutual_proximity_gammai_sparse(S:np.ndarray, 
                                    test_set_ind:np.ndarray=None, 
                                    verbose:int=0, log=None):
    """MP gammai for sparse similarity matrices. 
    
    Please do not directly use this function, but invoke via 
    mutual_proximity_gammai()
    """
    # TODO implement train_test split
    if test_set_ind is not None:
        raise NotImplementedError("MP gammai sparse does not support "
                                  "train/test split atm.")
    
    # mean, variance WITH zero values
    #=======================================================================
    # from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0  
    # mu, va = csr_mean_variance_axis0(self.S[train_set_mask])
    #=======================================================================
    
    # mean, variance WITHOUT zero values (missing values)
    mu = np.array(S.sum(0) / S.getnnz(0)).ravel()
    X = S.copy()
    X.data **= 2
    E1 = np.array(X.sum(0) / X.getnnz(0)).ravel()
    del X
    va = E1 - mu**2
    del E1
    
    A = (mu**2) / va
    B = va / mu
    del mu, va
    A[A < 0] = np.nan
    B[B <= 0] = np.nan

    S_mp = lil_matrix(S.shape, dtype=np.float32)
    n = S.shape[0]
    self_value = 1.
    
    for i in range(n):
        if verbose and log and ((i+1)%1000 == 0 or i+1 == n):
            log.message("MP_gammai: {} of {}".format(i+1, n), flush=True)
        j_idx = slice(i+1, n)
         
        Dij = S[i, j_idx].toarray().ravel() #Extract dense rows temporarily        
        p1 = _local_gamcdf(Dij, A[i], B[i])
        del Dij
        Dji = S[j_idx, i].toarray().ravel() #for vectorization below.
        p2 = _local_gamcdf(Dji, A[j_idx], B[j_idx])
        del Dji
        tmp = np.empty(n-i)
        tmp[0] = self_value / 2. 
        tmp[1:] = (p1 * p2).ravel()
        S_mp[i, i:] = tmp     
        del tmp, j_idx
    S_mp += S_mp.T
    
    return S_mp.tocsr()

def _local_gamcdf(x, a, b):
    """Gamma CDF, moment estimator"""
    try:
        a[a < 0] = np.nan
    except TypeError:
        if a < 0:
            a = np.nan
    try:
        b[b <= 0] = np.nan
    except TypeError:
        if b <= 0:
            b = np.nan
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
        """DEPRECATED"""
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        else:
            test_set_ind = None
        return mutual_proximity_gammai(self.D, 'similarity', test_set_ind, verbose)
    
    def mp_gammai(self, train_set_mask=None, verbose=False):
        """DEPRECATED"""
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance' 
        if train_set_mask is not None:
            test_set_ind = np.setdiff1d(np.arange(self.D.shape[0]), train_set_mask)
        else:#
            test_set_ind = None
        return mutual_proximity_gammai(self.D, metric, test_set_ind, verbose)
    
