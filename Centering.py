"""
Created on Oct 19, 2015

EARLY DEVELOPMENT VERSION

@author: Roman Feldbauer
"""

import numpy as np
from hub_toolbox import Distances as htd

class Centering(object):
    """Transform data (in vector space) by various 'centering' approaches."""


    def __init__(self, vectors:np.ndarray):
        """Create an object for subsequent centering of vector data."""
        
        self.vectors = np.copy(vectors)
        
    def centering(self):
        """Perform standard centering."""
        
        vectors_mean = np.mean(self.vectors, 0)
        vectors_cent = self.vectors - vectors_mean
        return vectors_cent
        
    def weighted_centering(self, gamma:float):
        """Perform weighted centering."""
        
        n = self.vectors.shape[0]
        d = np.zeros(n)
        vectors_sum = self.vectors.sum(0)
        for i in range(n):
            d[i] = n * np.inner(self.vectors[i], vectors_sum)
        d_sum = np.sum(d ** gamma)
        w = (d ** gamma) / d_sum
        vectors_mean_weighted = np.sum(w.reshape(n,1) * self.vectors, 0)
        vectors_weighted = self.vectors - vectors_mean_weighted
        return vectors_weighted
        
    def localized_centering(self, kappa:int, gamma:float = 1):
        """Perform localized centering."""
        # TODO CHECK CORRECTNESS!!
        
        if kappa == None:
            kappa = 20
            print("No 'kappa' defined for localized centering. \
                Using kappa=20 for 20 nearest neighbors.")
            
        # Rescale vectors to unit length
        v = self.vectors / np.sqrt((self.vectors ** 2).sum(-1))[..., np.newaxis]
        
        # for unit vectors it holds inner() == cosine()
        sim = -(htd.cosine_distance(v) - 1)
        n = sim.shape[0]
        local_affinity = np.zeros(n)
        for i in range(n):
            x = v[i]
            sim_i = sim[i, :]
            #TODO randomization
            nn = np.argsort(sim_i)[::-1][1 : kappa+1]
            c_kappa_x = np.mean(v[nn], 0)
            # c_kappa_x has not unit length in general
            local_affinity[i] = np.inner(x, c_kappa_x)       
            #local_affinity[i] = cosine(x, c_kappa_x) 
        sim_lcent = sim - (local_affinity ** gamma)
        return sim_lcent

if __name__ == '__main__':
    vectors = np.arange(12).reshape(3,4)
    c = Centering(vectors)
    print("Centering: ............. {}".format(c.centering()))
    print("Weighted centering: .... {}".format(c.weighted_centering(0.4)))
    print("Localized centering: ... {}".format(c.localized_centering(2)))