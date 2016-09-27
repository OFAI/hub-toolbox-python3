#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2015-2016, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import os
import sys
import numpy as np
from scipy import sparse
from hub_toolbox.Distances import cosine_distance

def load_dexter():
    """Load the example data set (dexter).
    
    Returns
    -------
    D : ndarray
        Distance matrix
    classes : ndarray
        Class label vector
    vectors : ndarray
        Vector data matrix
    """
        
    n = 300
    dim = 20000
    
    # Read class labels
    classes_file = os.path.dirname(os.path.realpath(__file__)) +\
        '/example_datasets/dexter_train.labels'
    classes = np.loadtxt(classes_file)  

    # Read data
    vectors = np.zeros((n, dim))
    data_file = os.path.dirname(os.path.realpath(__file__)) + \
        '/example_datasets/dexter_train.data'
    with open(data_file, mode='r') as fid:
        data = fid.readlines()       
    row = 0
    for line in data:
        line = line.strip().split() # line now contains pairs of dim:val
        for word in line:
            col, val = word.split(':')
            vectors[row][int(col)-1] = int(val)
        row += 1
    
    # Calc distance
    D = cosine_distance(vectors)
    return D, classes, vectors

def _check_distance_matrix_shape(D:np.ndarray):
    """ Check that matrix is quadratic. """
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")

def _check_distance_matrix_shape_fits_vectors(D:np.ndarray, vectors:np.ndarray):
    """ Check number of points in distance matrix equal number of vectors. """
    if D.shape[0] != vectors.shape[0]:
        raise TypeError("Data vectors dimension does not match "
                        "distance matrix (D) dimension.")

def _check_distance_matrix_shape_fits_labels(D:np.ndarray, classes:np.ndarray):
    """ Check the number of points in distance matrix equal number of labels"""
    if classes.size != D.shape[0]:
        raise TypeError("Number of class labels does not "
                        "match number of points.")

def _check_valid_metric_parameter(metric:str):
    """ Check parameter is either 'distance' or 'similarity'. """
    if metric != 'distance' and metric != 'similarity':
        raise ValueError("Parameter 'metric' must be "
                         "'distance' or 'similarity'.")

def copy_D_or_load_memmap(D, writeable=False): # pragma: no cover
    """Return a deep copy of a numpy array (if `D` is an ndarray), 
    otherwise return a read-only memmap (if `D` is a path).
    
    .. note:: Deprecated in hub-toolbox 2.3
              Will be removed in hub-toolbox 3.0.
              Memmap support will be dropped completely; individual 
              functions must deepcopy objects themselves.
    """
    print("DEPRECATED: memmap support will be dropped completely.", 
          file=sys.stderr)
    if isinstance(D, np.memmap):
        return D
    elif isinstance(D, np.ndarray):
        newD = np.copy(D.astype(np.float32))
    elif sparse.issparse(D):
        newD = D.copy()
    elif isinstance(D, str):
        if os.path.isfile(D):
            # keep matrix on disk
            if writeable:
                newD = np.load(D, mmap_mode='r+')
            else: # read-only
                newD = np.load(D, mmap_mode='r') 
        else:
            raise FileNotFoundError("Distance matrix file not found.")
    else:
        raise Exception("Distance matrix type not understood. "
                        "Must be np.ndarray or scipy.sparse.csr_matrix or "
                        "path to pickled ndarray.")
        
    return newD

def matrix_split(rows, cols, elem_size=8, nr_matrices=4): # pragma: no cover
    """Determine how to split a matrix that does not fit into memory. 
    
    Parameters
    ----------
    rows, cols : int 
        Shape of matrix that should be split.
    elem_size : int 
        memory requirement per matrix element in bytes. E.g. 8 bytes for float64
    nr_matrices : int 
        How many times must the split matrix fit into memory?
        This depends on the subsequent operations.
    
    Returns
    -------
    nr_batches : int
        number of submatrices
    nr_rows : int
        number of rows per submatrix.
    
    Notes
    -----
        - Submatrices always contain all columns per row. 
        - The last batch will usually have less rows than `nr_rows`
    """
    free_mem = FreeMemLinux(unit='k').user_free
    max_rows = int(free_mem / cols / elem_size)
    nr_rows = int(max_rows / nr_matrices)
    nr_batches = int(np.ceil(rows / nr_rows))
    return nr_batches, nr_rows

def random_sparse_matrix(size, density=0.05):
    """Generate a random sparse similarity matrix.
    
    Values are bounded by [0, 1]. Diagonal is all ones. The final density is
    approximately 2*`density`.
    
    Parameters
    ----------
    size : int
        Shape of the matrix (`size` x `size`)
    
    density : float, optional, default=0.05
        The matrix' density will be approximately 2 * `density`
        
    Returns
    -------
    S : csr_matrix
        Random matrix
    """
    S = sparse.rand(size, size, density, 'csr')
    S += S.T
    S /= S.max()
    S -= sparse.diags(S.diagonal(), 0)
    S += sparse.diags(np.ones(size), 0)
    return S

class FreeMemLinux(object): # pragma: no cover
    """Non-cross platform way to get free memory on Linux. 
    
    Original code by Oz123, 
    http://stackoverflow.com/questions/17718449/determine-free-ram-in-python
    """

    def __init__(self, unit='kB'):

        with open('/proc/meminfo', 'r') as mem:
            lines = mem.readlines()

        self._tot = int(lines[0].split()[1])
        self._free = int(lines[1].split()[1])
        self._buff = int(lines[2].split()[1])
        self._cached = int(lines[3].split()[1])
        self._shared = int(lines[20].split()[1])
        self._swapt = int(lines[14].split()[1])
        self._swapf = int(lines[15].split()[1])
        self._swapu = self._swapt - self._swapf

        self.unit = unit
        self._convert = self._factor()

    def _factor(self):
        """determine the conversion factor"""
        if self.unit == 'kB':
            return 1
        if self.unit == 'k':
            return 1024.0
        if self.unit == 'MB':
            return 1/1024.0
        if self.unit == 'GB':
            return 1/1024.0/1024.0
        if self.unit == '%':
            return 1.0/self._tot * 100
        else:
            raise Exception("Unit not understood")

    @property
    def total(self):
        return self._convert * self._tot

    @property
    def used(self):
        return self._convert * (self._tot - self._free)

    @property
    def used_real(self):
        """memory used which is not cache or buffers"""
        return self._convert * (self._tot - self._free - self._buff - self._cached)

    @property
    def shared(self):
        return self._convert * (self._tot - self._free)

    @property
    def buffers(self):
        return self._convert * (self._buff)

    @property
    def cached(self):
        return self._convert * self._cached

    @property
    def user_free(self):
        """This is the free memory available for the user"""
        return self._convert * (self._free + self._buff + self._cached)

    @property
    def swap(self):
        return self._convert * self._swapt

    @property
    def swap_free(self):
        return self._convert * self._swapf

    @property
    def swap_used(self):
        return self._convert * self._swapu
    
if __name__ == '__main__':
    fml = FreeMemLinux(unit='MB')
    fml2 = FreeMemLinux(unit='%')
    print("Used memory: {:.1f}M ({:.1f}%).".format(fml.used_real, fml2.used_real))
    print("Free memory: {:.1f}M ({:.1f}%).".format(fml.user_free, fml2.user_free))
