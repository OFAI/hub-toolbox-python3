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
import sys 
from enum import Enum

class Distribution(Enum):
    empiric = 'empiric'
    gauss = 'gauss'
    gaussi = 'gaussi'
    gammai = 'gammai'


class MutualProximity():
    """Transform a distance matrix with Mutual Proximity.
    
    """
    
    def __init__(self, D):
        self.D = np.copy(D)
        
    def calculate_mutual_proximity(self, distrType=None, test_set_mask=None):
        """Apply MP on a distance matrix."""
        
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)
            
        if distrType is None:
            print("No Mutual Proximity type given. Using: Distribution.empiric")
            print("For fast results use: Distribution.gaussi")
            Dmp = self.mp_empiric(train_set_mask)
        else:
            if distrType == Distribution.empiric:
                Dmp = self.mp_empiric(train_set_mask)
            elif distrType == Distribution.gauss:
                Dmp = self.mp_gauss(train_set_mask)
            elif distrType == Distribution.gaussi:
                Dmp = self.mp_gaussi(train_set_mask)
            elif distrType == Distribution.gammai:
                Dmp = self.mp_gammai(train_set_mask)
            else:
                self.warning("Valid Mutual Proximity type missing!\n"+\
                             "Use: \n"+\
                             "mp = MutualProximity(D, Distribution.empiric|"+\
                             "Distribution.gauss|Distribution.gaussi|"+\
                             "Distribution.gammi)\n"+\
                             "Dmp = mp.calculate_mutual_proximity()")
                Dmp = np.array([])
       
        return Dmp
         
    def mp_empiric_with_trainsetmask_BROKEN(self, train_set_mask=None):
        """Compute Mutual Proximity distances with empirical data (slow)."""
        
        #TODO implement train_set_mask! Currently BROKEN.
        np.fill_diagonal(self.D, 0)
        n = np.shape(self.D)[0]
        Dmp_list = [np.zeros(n-i) for i in range(n)]
        
        for i in range(n-1):
            
            # Select only finite distances for MP
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx, 0)
            
            dI = np.tile(self.D[i, train_set_mask], (j_len, 1))
            dJ = self.D[j_idx[:,None], train_set_mask]
            d = np.tile(self.D[j_idx, i][:, np.newaxis], (1, n))
            
            sIJ_intersect = np.sum((dI > d) & (dJ > d), 1)
            sIJ_overlap = 1 - (sIJ_intersect / n)
            Dmp_list[i] = sIJ_overlap
            
        Dmp = np.zeros(np.shape(self.D), dtype=self.D.dtype)
        for i in range(n-1):
            j_idx = np.arange(i+1, n)
            Dmp[i, j_idx] = Dmp_list[i]
            Dmp[j_idx, i] = Dmp_list[i]
            
        return Dmp
    
    def mp_empiric(self):
        """Compute Mutual Proximity distances with empirical data (slow)."""
         
        #TODO implement train_set_mask!
        np.fill_diagonal(self.D, 0)
        n = np.shape(self.D)[0]
        Dmp_list = [np.zeros(n-i) for i in range(n)]
         
        for i in range(n-1):
             
            # Select only finite distances for MP
            j_idx = np.arange(i+1, n)
            j_len = np.size(j_idx, 0)
             
            dI = np.tile(self.D[i, :], (j_len, 1))
            dJ = self.D[j_idx, :]
            d = np.tile(self.D[j_idx, i][:, np.newaxis], (1, n))
             
            sIJ_intersect = np.sum((dI > d) & (dJ > d), 1)
            sIJ_overlap = 1 - (sIJ_intersect / n)
            Dmp_list[i] = sIJ_overlap
             
        Dmp = np.zeros(np.shape(self.D), dtype=self.D.dtype)
        for i in range(n-1):
            j_idx = np.arange(i+1, n)
            Dmp[i, j_idx] = Dmp_list[i]
            Dmp[j_idx, i] = Dmp_list[i]
            
        return Dmp
    
    def mp_gauss(self, train_set_mask=None):
        """Compute Mutual Proximity distances with Gaussian model (very slow)."""
        
        np.fill_diagonal(self.D, 0)
        mu = np.mean(self.D[train_set_mask], 0)
        sd = np.std(self.D[train_set_mask], 0, ddof=1)
                
        #Code for the BadMatrixSigma error
        eps = np.spacing(1)
        epsmat = np.array([[1e5 * eps, 0], [0, 1e5 * eps]])
                
        Dmp = np.zeros(np.shape(self.D), dtype=self.D.dtype)
        n = np.size(self.D, 0)
        
        for i in range(n):
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
    
    def mp_gaussi(self, train_set_mask=None):
        """Compute Mutual Proximity modeled with independent Gaussians (fast)."""
        
        np.fill_diagonal(self.D, 0)
        mu = np.mean(self.D[train_set_mask], 0)
        sd = np.std(self.D[train_set_mask], 0, ddof=1)
        
        Dmp = np.zeros_like(self.D)
        n = np.size(self.D, 0)
        
        for i in range(n):
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
    
    def mp_gammai(self, train_set_mask=None):
        """Compute Mutual Proximity modeled with independent Gamma distributions."""
        
        np.fill_diagonal(self.D, 0)
        mu = np.mean(self.D[train_set_mask], 0)
        va = np.var(self.D[train_set_mask], 0, ddof=1)
        A = (mu**2) / va
        B = va / mu
        
        Dmp = np.zeros_like(self.D)
        n = np.size(self.D, 0)
        
        for i in range(n):
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
    
    def warning(self, *objs):
        print("WARNING: ", *objs, file=sys.stderr)
    