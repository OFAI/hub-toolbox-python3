"""
Intrinsic dimensionality estimation based on the DR-Toolbox 

This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7.2b.
The toolbox can be obtained from http://homepage.tudelft.nl/19j49
You are free to use, change, or redistribute this code in any way you
want for non-commercial purposes. However, it is appreciated if you 
maintain the name of the original author.

(C) Laurens van der Maaten, 2010
University California, San Diego / Delft University of Technology

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-15
"""
import numpy as np

class IntrinsicDim():
    """Calculate intrinsic dimensionality based on a MLE.
    
    
    """
       
    def __init__(self, X):
        # Deep copy required due to changes in vector data
        self.X = np.copy(X)
        
    def calculate_intrinsic_dimensionality(self):
        """Calculate intrinsic dimensionality based on a MLE."""
        
        # New array with unique rows                
        X = self.X[np.lexsort(np.fliplr(self.X).T)]  
        X -= np.tile(np.mean(X, 0), (np.size(X, 0), 1))
        X /= np.tile(np.var(X, 0) + 1e-7, (np.size(X, 0), 1))
        
        # Set neighborhood range to search in 
        k1 = 6
        k2 = 12
        
        # Compute matrix of log nearest neighbor distances
        X = X.T
        n = np.shape(X)[1]
        X2 = np.sum(X**2, 0)
        knnmatrix = np.zeros((k2, n))
        
        if n < 3000:
            distance = np.tile(X2, (n, 1)) + \
                np.tile(X2, (n,1)).T - 2 * np.dot(X.T, X) 
            distance = np.sort(distance, 0)
            # Replace invalid values with a small number 
            distance[distance<0] = 1e-7
            knnmatrix = .5 * np.log(distance[1:k2+1, :])
        else:
            for i in range(n):
                distance = np.sort(np.tile(X2[i], (1, n)) + X2 - 2 * np.dot(X[:, i], X) )
                # Replace invalid values with a small number 
                distance[distance<0] = 1e-7
                knnmatrix[:, i] = .5 * np.log(distance.ravel()[1:k2+1]).T 
        
        # Compute the ML estimate
        S = np.cumsum(knnmatrix, 0)
        k1k2range = np.arange(k1-1, k2)
        indexk = np.tile(k1k2range+1, (n, 1)).T
        dhat = -(indexk - 2) / ( S[k1-1:k2, :] - knnmatrix[k1-1:k2, :] * indexk)
        
        # Plot histogram of estimates for all datapoints
        # MATLAB: hist(mean(dhat), 80), pause
        
        # Average over estimates and over values of k
        no_dims = np.mean(dhat)
        return no_dims
