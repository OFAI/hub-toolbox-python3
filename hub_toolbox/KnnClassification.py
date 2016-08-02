# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2016, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import numpy as np
from scipy.sparse.base import issparse

class KnnClassification():
    """Perform k-nearest neighbor classification."""
    
    def __init__(self, D, classes, k, isSimilarityMatrix=False):
        if issparse(D):
            self.D = D
        else:
            self.D = np.copy(D)
        self.classes = np.copy(classes)
        if type(k) is np.ndarray:
            self.k = np.copy(k)
        else:
            self.k = np.array([k])
        self.isSimilarityMatrix = isSimilarityMatrix
        if self.isSimilarityMatrix:
            self.self_value = -np.inf
            self.sort_order = -1
        else:
            self.self_value = np.inf
            self.sort_order = 1
        assert D.shape[0] == len(classes)
        
    def perform_knn_classification(self):
        """Performs k-nearest neighbor classification.
        
        Use the distance matrix D (NxN) and the classes and perform a 
        k-NN experiment. Ties are broken by the nearest neighbor.
        The classification accuracy is returned in acc. 
        corr is a raw vector of the correctly classified items. 
        cmat is the confusion matrix.
        """
        
        # Why would there be a need for more than one k?
        k_length = np.size(self.k)
            
        acc = np.zeros( (k_length, 1) )
        corr = np.zeros( (self.D.shape[0], k_length) )
        
        n = np.size(self.D, 1)
        
        cl = np.sort(np.unique(self.classes))
        cmat = np.zeros( (len(cl), len(cl)) )
        
        classes = self.classes
        for idx in range(len(cl)):
            classes[self.classes == cl[idx]] = idx
            
        cl = range(len(cl))
        
        for i in range(n):
            seed_class = classes[i]
            
            if issparse(self.D):
                row = self.D[i, :].toarray().ravel()
            else:
                row = self.D[i, :]
            
            row[i] = self.self_value
            
            # Randomize, in case there are several points of same distance
            # (this is especially relevant for SNN rescaling)
            rp = np.indices( (np.size(self.D, 1), ) )[0]
            rp = np.random.permutation(rp)
            d2 = row[rp]
            d2idx = np.argsort(d2, axis=0)[::self.sort_order]
            idx = rp[d2idx]      
            
            # OLD code, non-randomized
            #idx = np.argsort(row)
            
            # More than one k?
            for j in range(k_length):
                nn_class = classes[idx[0:self.k[j]]] #smallest dist/highest sim
                cs = np.bincount(nn_class.astype(int))
                max_cs = np.where(cs == np.max(cs))[0]
                
                # "tie": use nearest neighbor
                if len(max_cs) > 1:
                    if seed_class == nn_class[0]:
                        acc[j] += 1/n 
                        corr[i, j] = 1
                    cmat[seed_class, nn_class[0]] += 1       
                # majority vote
                else:
                    if cl[max_cs] == seed_class:
                        acc[j] += 1/n
                        corr[i, j] = 1
                    cmat[seed_class, cl[max_cs]] += 1
                           
        return acc, corr, cmat
    
    def perform_knn_classification_with_test_set(self, test_set_mask=None):
        """Performs k-nearest neighbor classification.
        
        Use the distance matrix D (NxN) and the classes and perform a 
        k-NN experiment. Ties are broken by the nearest neighbor.
        The classification accuracy is returned in acc. 
        corr is a raw vector of the correctly classified items. 
        cmat is the confusion matrix.
        """        
        # Indices of training examples
        train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        # Why would there be a need for more than one k?
        k_length = np.size(self.k)
            
        acc = np.zeros( (k_length, 1) )
        corr = np.zeros( (self.D.shape[0], k_length) )
        
        # number of points to be classified
        #n = np.size(self.D, 1)
        n = np.size(test_set_mask)
        
        cl = np.sort(np.unique(self.classes))
        cmat = np.zeros( (len(cl), len(cl)) )
        
        classes = self.classes
        for idx in range(len(cl)):
            classes[self.classes == cl[idx]] = idx
            
        cl = range(len(cl))
        
        # Classify each point in test set
        #for i in range(n):
        for i in test_set_mask:
            seed_class = classes[i]
            
            if issparse(self.D):
                row = self.D[i, :].toarray().ravel()
            else:
                row = self.D[i, :]
            row[i] = self.self_value
            
            # Sort points in training set according to distance
            # Randomize, in case there are several points of same distance
            # (this is especially relevant for SNN rescaling)
            rp = train_set_mask
            rp = np.random.permutation(rp)
            d2 = row[rp]
            d2idx = np.argsort(d2, axis=0)[::self.sort_order]
            idx = rp[d2idx]      
            
            # OLD code, non-randomized
            #idx = np.argsort(row)
            
            # More than one k?
            for j in range(k_length):
                nn_class = classes[idx[0:self.k[j]]]
                cs = np.bincount(nn_class.astype(int))
                max_cs = np.where(cs == np.max(cs))[0]
                
                # "tie": use nearest neighbor
                if len(max_cs) > 1:
                    if seed_class == nn_class[0]:
                        acc[j] += 1/n 
                        corr[i, j] = 1
                    cmat[seed_class, nn_class[0]] += 1       
                # majority vote
                else:
                    if cl[max_cs] == seed_class:
                        acc[j] += 1/n
                        corr[i, j] = 1
                    cmat[seed_class, cl[max_cs]] += 1
                           
        return acc, corr, cmat