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

import numpy as np
from scipy import sparse
import os

def copy_D_or_load_memmap(D, writeable=False):
    """Return a deep copy of a numpy array (if D is an ndarray), 
    otherwise return a read-only memmap (if D is a path)."""
    
    if isinstance(D, np.memmap):
        return D
    elif isinstance(D, np.ndarray):
        newD = np.copy(D.astype(np.float32))
    elif sparse.issparse(D):
        #=======================================================================
        # log = Logging.ConsoleLogging()
        # log.warning("Not all classes of the hub toolbox support sparse matrices"
        #             " as of now. This is work-in-progress.")
        #=======================================================================
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

def matrix_split(rows, cols, elem_size=8, nr_matrices=4):
    """Determine how to split a matrix that does not fit into memory. 
    
    Parameters:
    -----------
    rows, cols  ... Shape of matrix that should be split.
    elem_size   ... memory requirement per matrix element in bytes. E.g. 8 bytes for float64
    nr_matrices ... How many times must the split matrix fit into memory?
                    This depends on the subsequent operations. \n
    Returns:
    --------
    nr_batches ... number of submatrices
    nr_rows    ... number of rows per submatrix. \n
    Notes: 
    1) Submatrices always contain all columns per row. 
    2) The last batch will usually have less rows than nr_rows
    """
    free_mem = FreeMemLinux(unit='k').user_free        
    max_rows = int(free_mem / cols / elem_size) 
    nr_rows = int(max_rows / nr_matrices) 
    nr_batches = int(np.ceil(rows / nr_rows))
    return nr_batches, nr_rows

class FreeMemLinux(object):
    """
    Non-cross platform way to get free memory on Linux. 
    
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
            return 1.0/self._tot
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
