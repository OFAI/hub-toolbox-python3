#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2015-2017, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import os
import numpy as np
from scipy import sparse
from scipy.sparse.base import issparse

__all__ = ['load_dexter', 'random_sparse_matrix', 
           'load_csr_matrix', 'save_csr_matrix']

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
    from hub_toolbox.Distances import cosine_distance
        
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

def check_is_nD_array(arr:np.ndarray, n:int, arr_type=''):
    """ Check that array is exactly n dimensional. """
    try:
        if arr.ndim != n:
            raise TypeError(arr_type + " array must be a " + str(n) +
                            "D array, but was found to be a " +
                            str(arr.ndim) + "D array with shape: " +
                            str(arr.shape))
    except AttributeError:
        raise TypeError("Object 'arr' does not seem to be an array.")

def check_distance_matrix_shape(D:np.ndarray):
    """ Check that matrix is quadratic. """
    check_is_nD_array(D, n=2, arr_type="Distance/similarity")
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic. "
                        "Shape: {}".format(D.shape))

def check_distance_matrix_shape_fits_vectors(D:np.ndarray, vectors:np.ndarray):
    """ Check number of points in distance matrix equal number of vectors. """
    check_is_nD_array(D, 2, "Distance/similarity")
    check_is_nD_array(vectors, 2, "Data vectors")
    if D.shape[0] != vectors.shape[0]:
        raise TypeError("Number of points in `vectors` does not match "
                        "number of points in `D`. Shape of `vectors`: {}, "
                        "shape of `D`: {}".format(vectors.shape[0], D.shape[0]))

def check_distance_matrix_shape_fits_labels(D:np.ndarray, classes:np.ndarray):
    """ Check the number of points in distance matrix equal number of labels."""
    check_is_nD_array(D, 2, "Distance/similarity")
    check_is_nD_array(classes, 1, "Class label")
    if classes.size != D.shape[0]:
        raise TypeError("Number of class labels does not match number of "
                        "points. Labels: {}, points: {}."
                        .format(classes.size, D.shape[0]))

def check_vector_matrix_shape_fits_labels(X:np.ndarray, classes:np.ndarray):
    """ Check the number of points in vector matrix equal number of labels."""
    check_is_nD_array(X, 2, "Data vectors")
    check_is_nD_array(classes, 1, "Class label")
    if classes.size != X.shape[0]:
        raise TypeError("Number of class labels does not match number of "
                        "points. Labels: {}, points: {}."
                        .format(classes.size, X.shape[0]))

def check_sample_shape_fits(D:np.ndarray, idx:np.ndarray):
    """ Check that number of columns in ``D`` equals the size of ``idx``. """
    if issparse(D) or issparse(idx):
        raise TypeError("Sparse matrices are not supported for SampleMP.")
    check_is_nD_array(D, 2, "Distance/similarity")
    check_is_nD_array(idx, 1, "Index")
    if D.shape[1] > D.shape[0]:
        raise ValueError("Number of samples is higher than number of points. "
                         "Must be less than or equal. In the latter case, "
                         "consider not using samples at all for efficiency. "
                         "Shape of `D`: {}.".format(D.shape))
    if D.shape[1] != idx.size:
        raise TypeError("Number of samples in index array does not match "
                        "the number of samples in the data matrix. "
                        "Size of `idx`: {}, Columns in `D`: {}."
                        .format(idx.size, D.shape[1]))

def check_valid_metric_parameter(metric:str):
    """ Check parameter is either 'distance' or 'similarity'. """
    if metric != 'distance' and metric != 'similarity':
        raise ValueError("Parameter 'metric' must be "
                         "'distance' or 'similarity'."
                         "Got: " + metric.__str__())

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

def save_csr_matrix(file, matrix):
    np.savez(file, data=matrix.data, indices=matrix.indices,
             indptr=matrix.indptr, shape=matrix.shape)
    return file

def load_csr_matrix(file):
    container = np.load(file)
    return sparse.csr_matrix((container['data'], container['indices'], 
                              container['indptr']), shape=container['shape'])

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
