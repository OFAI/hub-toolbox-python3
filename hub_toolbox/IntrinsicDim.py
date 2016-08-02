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

This file is based on a Matlab script by Elizaveta Levina, University of 
Michigan, available at http://dept.stat.lsa.umich.edu/~elevina/mledim.m

Reference:  E. Levina and P.J. Bickel (2005).  
 "Maximum Likelihood Estimation  of Intrinsic Dimension."  
 In Advances in NIPS 17, Eds. L. K. Saul, Y. Weiss, L. Bottou. 
"""
import numpy as np
import sys

class IntrinsicDim():
    """Calculate intrinsic dimensionality based on a MLE."""
       
    def __init__(self, X, data_type='vector'):
        """ 
        Estimate intrinsic dimensionality in vector, distance, or similarity
        data with data_type=['vector', 'distance', 'similarity'], respectively.
        
        Please note that the MLE was derived for euclidean distances. Using 
        other (dis)similarity measures may lead to undefined results.
        """
        # Deep copy required due to changes in vector data
        self.X = X.copy()
        if data_type in ['vector', 'distance', 'similarity']:
            self.data_type = data_type
            if data_type != 'vector':
                raise NotImplementedError("IntrinsicDim currently only "
                                          "supports vector data.")
        else:
            raise ValueError("Parameter data_type must be 'vector', 'distance'"
                             " , or 'similarity'. Got '{}' instead.".
                             format(data_type.__str__())) 

    def calculate_intrinsic_dimensionality(self, k1=6, k2=12, 
                                           estimator='levina'):
        """Calculate intrinsic dimensionality based on a MLE.
        
        Parameters k1 and k2 determine the neighborhood range to search in
        (default: k1=6, k2=12).
        
        Parameter 'estimator' determines the summation strategy: 'levina' 
        (default) or 'mackay' (see http://www.inference.phy.cam.ac.uk/
        mackay/dimension/)."""
        
        n = self.X.shape[0]
        
        if estimator not in ['levina', 'mackay']:
            raise ValueError("Unknown estimator '{}', please use 'levina' or "
                             "'mackay' instead.".format(estimator.__str__()))
        if k1 < 1 or k2 < k1 or k2 >= n:
            raise ValueError("Invalid neighborhood: Please make sure that "
                             "0 < k1 <= k2 < n. (Got k1={} and k2={}).".
                             format(k1, k2))
        
        if self.data_type == 'vector':
            # New array with unique rows                
            X = self.X[np.lexsort(np.fliplr(self.X).T)]  
            del self.X # allow to free memory
            
            # Standardization
            X -= X.mean(0) # broadcast
            X /= X.var(0) + 1e-7 # broadcast
        
            # Compute matrix of log nearest neighbor distances
            X2 = (X**2).sum(1)
        
            if n <= 5000: # speed-memory trade-off
                distance = X2.reshape(-1, 1) + X2 - 2*np.dot(X, X.T) #2x br.cast 
                distance.sort(1)
                # Replace invalid values with a small number 
                distance[distance<0] = 1e-7
                knnmatrix = .5 * np.log(distance[:, 1:k2+1])
            else:
                knnmatrix = np.zeros((n, k2))
                for i in range(n):
                    distance = np.sort(X2[i] + X2 - 2 * np.dot(X, X[i, :]))
                    # Replace invalid values with a small number 
                    distance[distance<0] = 1e-7
                    knnmatrix[i, :] = .5 * np.log(distance[1:k2+1]) 
        elif self.data_type == 'distance':
            # TODO calculation WRONG
            self.X.sort(1)
            self.X[self.X < 0] = 1e-7
            knnmatrix = np.log(self.X[:, 1:k2+1])
        elif self.data_type == 'similarity':
            # TODO calculation WRONG
            print("WARNING: using similarity data may return "
                  "undefined results.", file=sys.stderr)
            self.X[self.X < 0] = 0
            distance = 1 - (self.X / self.X.max())
            knnmatrix = np.log(distance[:, 1:k2+1])
        
        # Compute the ML estimate
        S = np.cumsum(knnmatrix, 1)
        indexk = np.arange(k1, k2+1) # broadcasted afterwards
        dhat = -(indexk - 2) / (S[:, k1-1:k2] - knnmatrix[:, k1-1:k2] * indexk)
           
        if estimator == 'levina':  
            # Average over estimates and over values of k
            no_dims = dhat.mean()
        elif estimator == 'mackay':
            # Average over inverses
            dhat **= -1
            dhat_k = dhat.mean(0)
            no_dims = (dhat_k ** -1).mean()
                     
        return int(no_dims.round())
