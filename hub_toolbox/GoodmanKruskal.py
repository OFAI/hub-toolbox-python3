"""
Computes the Goodman-Kruskal clustering index.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>
(c) 2016, Roman Feldbauer <roman.feldbauer@ofai.at>

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
     
@author: Roman Feldbauer
@date: 2015-09-18
"""

import numpy as np

def goodman_kruskal_index(D:np.array, classes:np.array,
                          metric='distance') -> float:
    """Calculate the Goodman-Kruskal clustering index.
    
    This clustering quality measure relates the number of concordant (Q_c) 
    and discordant (Q_d) quadruples (d_ij, d_kl) of a distance matrix.
    We only consider tuples, so that i, j are from the same class 
    and k, l are from different classes. Then a quadruple is...
    concordant, if d_ij < d_kl
    discordant, if d_ij > d_kl
    non counted, otherwise.
    
    The Goodman-Kruskal index gamma is defined as: 
        gamma = (Q_c - Q_d) / (Q_c + Q_d)
        
    gamma is bounded to [-1, 1], where larger values indicate better clustering.
    
    Parameters:
    -----------
    D : np.array
        The n x n symmetric distance (similarity) matrix.
    
    classes : np.array
        The 1 x n vector of class labels for each point.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether the matrix 'D' is a distance or similarity matrix

    Returns:
    --------
    gamma : float
        Goodman-Kruskal index in [-1, 1] (higher=better)
    """
    
    # Checking input
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if classes.size != D.shape[0]:
        raise TypeError("Number of class labels does not match number of points.")
    if metric == 'distance':
        sort_order = 1
    elif metric == 'similarity':
        sort_order = -1
    else:
        raise ValueError("Parameter 'metric' must be 'distance' or 'similarity'.")
    
    # Calculations
    Q_c = 0.0
    Q_d = 0.0
    cls = np.unique(classes)
    
    # D_kl pairs in different classes
    other = classes[:, np.newaxis] != classes[np.newaxis, :]
    D_other = D[np.triu(other, 1)]
        
    for c in cls:
        sel = classes == c 
        if np.sum(sel) > 1: 
            sel = sel[:, np.newaxis].astype(bool)
            selD = np.logical_and(sel, sel.T)  
            # D_ij pairs within same class
            D_self = D[np.triu(selD, 1).astype(bool).T].T
        else:
            # skip if there is only one item per class
            continue
        # D_kl pairs in different classes (D_other) are computed once for all c
        D_full = np.append(D_self, D_other)

        self_size = np.max(np.shape(D_self))
        other_size = np.max(np.shape(D_other))
        # Sort algorithm must be stable!
        full_idx = np.argsort(D_full, kind='mergesort')[::sort_order]
        
        # Calc number of quadruples with equal distance
        n_equidistant = 0
        sdf = np.sort(D_full, axis=None)
        equi_mask = np.zeros(sdf.size, dtype=bool)
        # Positions with repeated values
        equi_mask[1:] = sdf[1:] == sdf[:-1]
        equi_dist = sdf[equi_mask]
        # How often does each value occur in self/other:
        for dist in np.unique(equi_dist):
            equi_arg = np.where(D_full == dist)[0]
            self_equi = (equi_arg < self_size).sum()
            other_equi = len(equi_arg) - self_equi
            # Number of dc that are actually equal
            n_equidistant += self_equi * other_equi

        # Calc number of concordant quadruples
        cc = 0
        ccsize = other_size
        for idx in full_idx:
            if idx < self_size:
                cc += ccsize
            else:
                ccsize -= 1

        # Calc number of discordant quadruples
        dc = self_size * other_size - cc - n_equidistant

        Q_c += cc
        Q_d += dc
    
    # Calc Goodman-Kruskal's gamma
    if Q_c + Q_d == 0:
        gamma = 0.0
    else:
        gamma = (Q_c - Q_d) / (Q_c + Q_d)

    return gamma 

# DEPRECATED class GoodmanKruskal. Remove for next hub_toolbox release.
class GoodmanKruskal():
    
    def __init__(self, D, classes, isSimilarityMatrix=False):
        self.D = D
        self.classes = classes
        self.is_similarity_matrix = isSimilarityMatrix
        
    def calculate_goodman_kruskal_index(self) -> float:
        """Calculate the Goodman-Kruskal clustering index."""
        import sys
        print("DEPRECATED: Please use GoodmanKruskal.goodman_kruskal_index "
              "instead.", file=sys.stderr)
        if self.is_similarity_matrix:
            metric = 'similarity'
        else:
            metric = 'distance'
        return goodman_kruskal_index(self.D, self.classes, metric)