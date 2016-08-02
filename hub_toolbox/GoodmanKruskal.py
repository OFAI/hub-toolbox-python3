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
from scipy.sparse import csr_matrix, lil_matrix

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
    if metric != 'distance' and metric != 'similarity':
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
        full_idx = np.argsort(D_full, kind='mergesort')[::-1]
        
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
        if metric == 'similarity':
            gamma = (Q_c - Q_d) / (Q_c + Q_d)
        else:
            gamma = (Q_d - Q_c) / (Q_c + Q_d)

    return gamma 

def sparse_goodman_kruskal_index(S:csr_matrix, classes:np.ndarray, 
                                 metric='similarity') -> float:
    """Calculate the Goodman-Kruskal clustering index.
    
    This clustering quality measure relates the number of concordant (Q_c) 
    and discordant (Q_d) quadruples (s_ij, s_kl) of a similarity matrix.
    We only consider tuples, so that i, j are from the same class 
    and k, l are from different classes. Then a quadruple is...
    concordant, if s_ij > s_kl
    discordant, if s_ij < s_kl
    non counted, otherwise.
    
    The Goodman-Kruskal index gamma is defined as: 
        gamma = (Q_c - Q_d) / (Q_c + Q_d)
        
    gamma is bounded to [-1, 1], where larger values indicate better 
    clustering.
    
    Parameters:
    -----------
    S : csr_matrix
        The n x n symmetric similarity matrix.
    
    classes : np.array
        The 1 x n vector of class labels for each point.
    
    metric : {'similarity', 'distance'}, optional (default: 'similarity')
        Define, whether the matrix 'D' is a distance or similarity matrix.
        NOTE: 'distance' is used for debugging purposes only. Use standard
              goodman_kruskal_index function for distance matrices.

    Returns:
    --------
    gamma : float
        Goodman-Kruskal index in [-1, 1] (higher=better)
    """
    
    # Checking input
    if S.shape[0] != S.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if classes.size != S.shape[0]:
        raise TypeError("Number of class labels does not match "
                        "number of points.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")
        
    # Calculations
    Qc = 0.0
    Qd = 0.0
    n = classes.size
    
    # S_kl pairs in different classes
    S_other_list = lil_matrix((n, n))
    other_nnz = 0
    # building the complete mask at once would result in dense N x N matrix
    for i, c in enumerate(classes):
        cur_other = csr_matrix((c != classes)[i+1:])
        other_nnz += cur_other.nnz
        S_other_list[i, :cur_other.shape[1]] = \
            S[i, i+1:].multiply(cur_other)
    n_other_zeros = other_nnz - S_other_list.nnz
    # The following might be achieved faster w/o csr intermediate
    S_other = S_other_list.tocsr().data
    del S_other_list, cur_other
    
    cls = np.unique(classes)
    for c in cls:
        sel = classes == c 
        if np.sum(sel) > 1: 
            n = sel.size
            # intra-class distances
            S_self_list = lil_matrix((n, n))
            self_nnz = 0
            for i, s in enumerate(sel):
                cur_self = csr_matrix((s * sel)[i+1:])
                self_nnz += cur_self.nnz
                S_self_list[i, :cur_self.shape[1]] = \
                    S[i, i+1:].multiply(cur_self)
            n_self_zeros = self_nnz - S_self_list.nnz
            # Same as with S_other
            S_self = S_self_list.tocsr().data
            del S_self_list, cur_self
        else:
            # skip if there is only one item per class
            continue
        
        # S_kl pairs in different classes (S_other) are computed once for all c
        S_full_data = np.append(S_self, S_other)

        self_data_size = S_self.size
        self_size = S_self.size + n_self_zeros
        other_size = S_other.size + n_other_zeros
        full_data_idx = np.argsort(S_full_data, kind='mergesort')[::-1] 
        del S_self

        # Calc number of quadruples with equal distance
        n_equidistant = 0
        sdf = np.sort(S_full_data, axis=None)
        equi_mask = np.zeros(sdf.size, dtype=bool)
        # Positions with repeated values
        equi_mask[1:] = sdf[1:] == sdf[:-1]
        equi_dist = sdf[equi_mask]
        # How often does each value occur in self/other:
        for dist in np.unique(equi_dist):
            equi_arg = np.where(S_full_data == dist)[0]
            self_equi = (equi_arg < self_data_size).sum()
            other_equi = len(equi_arg) - self_equi
            # Number of dc that are actually equal
            n_equidistant += self_equi * other_equi
        del S_full_data
        n_zero = n_self_zeros * n_other_zeros
        
        # Calc number of concordant quadruples
        cc = 0
        ccsize = other_size
        #ccsize = other_size
        for idx in full_data_idx:
            if idx < self_data_size:
                cc += ccsize
            else:
                ccsize -= 1
        del full_data_idx
        
        # Calc number of discordant quadruples
        dc = self_size * other_size - cc - n_equidistant - n_zero
        Qc += cc
        Qd += dc

    # Calc Goodman-Kruskal's gamma
    if Qc + Qd == 0:
        di = 0.0
    else:
        if metric == 'similarity':
            di = (Qc - Qd) / (Qc + Qd)
        elif metric == 'distance':
            di = (Qd - Qc) / (Qc + Qd)
        else:
            import sys
            print("WARNING: Unknown metric type {}. Assuming 'similarity' "
                  "instead. Sign of result might be reversed, if this is "
                  "wrong!".format(metric.__str__[0:32]), file=sys.stderr)
            di = (Qc - Qd) / (Qc + Qd)
    return di 

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