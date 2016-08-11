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
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

def goodman_kruskal_index(D:np.ndarray, classes:np.ndarray,
                          metric:str='distance') -> float:
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
    D : ndarray
        The n x n symmetric distance (similarity) matrix.
    
    classes : ndarray
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
                                 metric='similarity', zero_mv:bool=False, 
                                 heuristic:str=None, verbose:int=0) -> float:
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
    
    classes : ndarray
        The 1 x n vector of class labels for each point.
    
    metric : {'similarity', 'distance'}, optional (default: 'similarity')
        Define, whether the matrix 'D' is a distance or similarity matrix.
        NOTE: 'distance' is used for debugging purposes only. Use standard
              goodman_kruskal_index function for distance matrices.
              
    zero_mv : boolean, optional (default: False)
        Treat zeros as missing values, i.e. tuples with any zero
        similarities are not counted.
        
    heuristic : {None, 'equal_sim'}, optional (default: None)
        * None - Exact GK
        * 'equal_sim' - omit expensive search for equal similarities
                        Useful, when no/few equal similarites are expected.
                        Do NOT use in case of SharedNN matrices!
                        NOTE: Equal zero similarities are still considered
                              when using this heuristic.
    
    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

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
    if verbose:
        print("Sparse Goodman-Kruskal")
        sys.stdout.write("----------------------")
        print(flush=True)
    # Calculations
    Qc = 0.0
    Qd = 0.0
    n = classes.size
    
    # S_kl pairs in different classes
    S_other_list = lil_matrix((n, n))
    other_nnz = 0
    # building the complete mask at once would result in dense N x N matrix
    if verbose >= 2:
        print("Finding S_kl pairs with different class labels...", 
              end=' ', flush=True)
    for i, c in enumerate(classes):
        cur_other = csr_matrix((c != classes)[i+1:])
        other_nnz += cur_other.nnz
        S_other_list[i, :cur_other.shape[1]] = \
            S[i, i+1:].multiply(cur_other)
    n_other_zeros = other_nnz - S_other_list.nnz
    # The following might be achieved faster w/o csr intermediate
    S_other = S_other_list.tocsr().data
    del S_other_list, cur_other
    if verbose >= 2:
        print("done.", flush=True)
    
    cls = np.unique(classes)
    for c in cls:
        if verbose == 1:# and c % 10 == 0:
            # end='\r' does not work with jupyter notebook
            print("Class: {}/{}".format(c, len(cls)), end='')
        sel = classes == c
        if np.sum(sel) > 1:
            if verbose >= 2:
                print("Finding S_ij pairs for class {}..."
                      .format(c), end=' ')
            n = sel.size
            # intra-class distances
            S_self_list = lil_matrix((n, n))
            self_nnz = 0
            
            # Only visit points of self class
            sel_arg = np.where(sel > 0)[0]
            for i in sel_arg:
                cur_self = csr_matrix(sel[i+1:])
                self_nnz += cur_self.nnz
                S_self_list[i, :cur_self.shape[1]] = \
                    S[i, i+1:].multiply(cur_self)
            
            n_self_zeros = self_nnz - S_self_list.nnz
            # Same as with S_other
            S_self = S_self_list.tocsr().data
            del S_self_list, cur_self
            if verbose >= 2:
                print("done.")
        else:
            # skip if there is only one item per class
            if verbose == 1: # and c % 10 == 0:
                sys.stdout.write('\r')
            continue
        
        # S_kl pairs in different classes are computed once for all c
        if verbose >= 2:
            print("Sorting data...", end=' ')
        S_full_data = np.append(S_self, S_other)
        
        self_data_size = S_self.size
        self_size = S_self.size + n_self_zeros
        other_data_size = S_other.size
        other_size = S_other.size + n_other_zeros
        full_data_idx = np.argsort(S_full_data, kind='mergesort')[::-1]
        del S_self
        if verbose >= 2:
            print("done.", flush=True)
        
        # Calc number of quadruples with equal distance
        if verbose >= 2:
            print("Calculating number of quadruples with equal distance...", 
                  end=' ')
        n_equidistant = 0
        # Number of equal zero similarities
        if zero_mv:
            n_zero = 0
        else:
            n_zero = n_self_zeros * n_other_zeros
        if heuristic == 'equal_sim':
            if verbose >= 2:
                print("OMITTED (heuristic).")
            else:
                pass
        else:
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
            del S_full_data, equi_mask, equi_dist, equi_arg
            if verbose >= 2:
                print("done.", flush=True)
        
        # Calc number of concordant quadruples
        if verbose >= 2:
            print("Calculating number of concordant quadruples...", end=' ')
        cc = 0
        if zero_mv:
            ccsize = other_data_size
        else:
            ccsize = other_size
        for idx in full_data_idx:
            if idx < self_data_size:
                cc += ccsize
            else:
                ccsize -= 1
        if verbose >= 2:
            print("done.", flush=True)
        
        # Calc number of discordant quadruples
        if verbose >= 2:
            print("Calculating number of discordant quadruples...", end=' ')
        if zero_mv:
            dc = self_data_size * other_data_size - cc - n_equidistant
        else:
            dc = self_size * other_size - cc - n_equidistant - n_zero
        Qc += cc
        Qd += dc
        if verbose >= 2:
            print("done.", flush=True)
        if verbose == 1: # and c % 10 == 0:
            sys.stdout.write('\r')
    
    # Calc Goodman-Kruskal's gamma
    if verbose >= 2:
        print("Calculating Goodman-Kruskal gamma...", end=' ')
    if Qc + Qd == 0:
        gamma = 0.0
    else:
        if metric == 'similarity':
            gamma = (Qc - Qd) / (Qc + Qd)
        elif metric == 'distance':
            gamma = (Qd - Qc) / (Qc + Qd)
        else:
            print("WARNING: Unknown metric type {}. Assuming 'similarity' "
                  "instead. Sign of result might be reversed, if this is "
                  "wrong!".format(metric.__str__[0:32]), file=sys.stderr)
            gamma = (Qc - Qd) / (Qc + Qd)
    if verbose >= 2:
        print("done.", flush=True)
    return gamma

# DEPRECATED class GoodmanKruskal. Remove for next hub_toolbox release.
class GoodmanKruskal():
    
    def __init__(self, D, classes, isSimilarityMatrix=False):
        self.D = D
        self.classes = classes
        self.is_similarity_matrix = isSimilarityMatrix
        
    def calculate_goodman_kruskal_index(self) -> float:
        """Calculate the Goodman-Kruskal clustering index."""
        print("DEPRECATED: Please use GoodmanKruskal.goodman_kruskal_index "
              "instead.", file=sys.stderr)
        if self.is_similarity_matrix:
            metric = 'similarity'
        else:
            metric = 'distance'
        return goodman_kruskal_index(self.D, self.classes, metric)
