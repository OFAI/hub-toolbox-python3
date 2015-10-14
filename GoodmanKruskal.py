"""
Computes the Goodman-Kruskal clustering index.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

The Goodman-Kruskal index is a
clustering quality measure that relates the number of concordant ($Q_c$) and
discordant (Q_d) tuples (d_{i,j}, d_{k,l}) of a distance matrix.
 * A tuple is concordant if its items i, j are from the same class,
   items k, l are from different classes and d_{i,j} < d_{k,l}.
 * A tuple is discordant if its items i, j are from the same class,
   items k, l are from different classes and d_{i,j} > d_{k,l}.
 * A tuple is not counted if it is neither concordant nor discordant,
   that is, if d_{i,j} = d_{k,l}.

The Goodman-Kruskal Index ($I_{GK}$) is defined as:
I_{GK} = \frac{Q_c - Q_d}{Q_c + Q_d}.

I_{GK} is bounded to the interval [-1, 1], and the higher I_{GK}, the
more concordant and fewer discordant quadruples are present in the data set.
Thus a large index value indicates a good clustering (in terms of
pairwise stability.

Usage:
  goodman_kruskal(D, classes) - Where D is an NxN distance matrix and
     classes is a vector with the class labels as integers.
     
This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-18
"""

import numpy as np

class GoodmanKruskal():
    
    def __init__(self, D, classes):
        self.D = np.copy(D)
        self.classes = np.copy(classes)
        
    def calculate_goodman_kruskal_index(self) -> float:
        """Calculate the Goodman-Kruskal clustering index."""
        Qc = 0.0
        Qd = 0.0
        cls = np.unique(self.classes)
        for c in cls:
            sel = self.classes == c 
            if np.sum(sel) > 1: # True encoded as 1
                #selD[(sel, sel)] = True # how can this be achieved in numpy?
                # DGEMM does the job, not as fast as MATLAB code, but
                # still MUCH faster than looping through selD
                sel = sel[:, np.newaxis]
                selD = np.dot(sel, sel.T).astype(bool)    
                #selD = np.zeros(np.shape(self.D)).astype(bool)
                #selD[sel, sel] = True # python selD[sel,sel] != matlab selD(sel,sel)!!
                D_self = self.D[np.triu(selD, 1).astype(bool).T].T#.ravel()
            else:
                # skip if there is only one item per class
                continue
            # again, not as efficient as matlab
            not_sel = ~sel.astype(bool)
            other = np.dot(sel, not_sel.T)
            D_other = self.D[other.T]
            D_full = np.append(D_self, D_other)
            
            self_size = np.max(np.shape(D_self))
            other_size = np.max(np.shape(D_other))
            
            # Randomize, in case there are several points of same distance
            # (this is especially relevant for SNN rescaling)
            rp = np.random.permutation(np.size(D_full))
            d2 = D_full[rp]
            d2idx = np.argsort(d2)
            full_idx = rp[d2idx]      
            
            # OLD code, non-randomized
            #full_idx = np.argsort(D_full)
            
            cc = 0
            ccsize = other_size
            for idx in full_idx:
                if idx < self_size:
                    cc += ccsize
                else:
                    ccsize -= 1
            
            dc = self_size * other_size - cc
            
            Qc += cc
            Qd += dc
        
        if Qc + Qd == 0:
            di = 0.0
        else:
            di = (Qc - Qd) / (Qc + Qd)
        
        return di 