"""
Created on Oct 19, 2015

EARLY DEVELOPMENT VERSION

@author: Roman Feldbauer
"""

import numpy as np
from hub_toolbox.Distances import cosine_distance, Distance, euclidean_distance

class Centering(object):
    """Transform data (in vector space) by various 'centering' approaches."""


    def __init__(self, vectors:np.ndarray, is_distance_matrix=False):
        """Create an object for subsequent centering of vector data. \
        Set is_distance_matrix=True when using distance data.
        """
        self.vectors = np.copy(vectors)
        self.distance_matrix = is_distance_matrix
                
    def centering(self, distance_based=False, test_set_mask=None):
        """Perform standard centering.
        
        Returns centered vectors (not distance matrix!)
        """
        
        if test_set_mask is not None:
            if distance_based:
                raise NotImplementedError("Distance based centering does not "
                                          "support train/test splits so far.")
            train_set_mask = np.setdiff1d(\
                np.arange(self.vectors.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.vectors.shape[0], np.bool)
            
        if distance_based:
            n = self.vectors.shape[0]
            H = np.identity(n) - (1.0/n) * np.ones((n, n))
            K = self.vectors
            K_cent = H.dot(K).dot(H)
            return K_cent
        else:
            vectors_mean = np.mean(self.vectors[train_set_mask], 0)
            vectors_cent = self.vectors - vectors_mean
            return vectors_cent
        
    def weighted_centering(self, gamma, 
                           distance_metric=Distance.cosine, test_set_mask=None):
        """Perform weighted centering.
        
        Returns a distance matrix (not centered vectors!)
        """
                   
        # Indices of training examples
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(\
                np.arange(self.vectors.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.vectors.shape[0], np.bool)
        
        n = self.vectors.shape[0]
        n_train = self.vectors[train_set_mask].shape[0]
        d = np.zeros(n)
        
        if distance_metric == Distance.cosine:
            vectors_sum = self.vectors[train_set_mask].sum(0)
            for i in np.arange(n):
                #d[i] = n_train * np.inner(self.vectors[i], vectors_sum / n_train)
                #d[i] = n_train * cosine(self.vectors[i], vectors_sum / n_train)
                d[i] = n_train * cosine_distance(\
                        np.array([self.vectors[i], vectors_sum/n_train]))[0, 1]
        elif distance_metric == Distance.euclidean:
            for i in range(n):
                displ_v = self.vectors[train_set_mask] - d[i]
                d[i] = np.sum(np.sqrt(displ_v * displ_v))
        else:
            raise ValueError("Weighted centering currently only supports "
                             "cosine and euclidean distances.")
        d_sum = np.sum(d ** gamma)
        w = (d ** gamma) / d_sum
        vectors_mean_weighted = np.sum(w.reshape(n,1) * self.vectors, 0)
        vectors_weighted = self.vectors - vectors_mean_weighted
        return vectors_weighted
        
    #===========================================================================
    # def localized_centering(self, kappa:int, gamma:float=1, 
    #                         distance_metric='cosine',test_set_mask=None):
    #     """Perform localized centering.
    #     
    #     Returns a distance matrix (not centered vectors!)
    #     """
    #     # TODO CHECK CORRECTNESS!!
    #     
    #     if kappa == None:
    #         kappa = 20
    #         print("No 'kappa' defined for localized centering. \
    #             Using kappa=20 for 20 nearest neighbors.")
    #         
    #     # Rescale vectors to unit length
    #     v = self.vectors / np.sqrt((self.vectors ** 2).sum(-1))[..., np.newaxis]
    #     
    #     # for unit vectors it holds inner() == cosine()
    #     sim = -(htd.cosine_distance(v) - 1)
    #     n = sim.shape[0]
    #     local_affinity = np.zeros(n)
    #     for i in range(n):
    #         x = v[i]
    #         sim_i = sim[i, :]
    #         #TODO randomization
    #         nn = np.argsort(sim_i)[::-1][1 : kappa+1]
    #         c_kappa_x = np.mean(v[nn], 0)
    #         # c_kappa_x has no unit length in general
    #         local_affinity[i] = np.inner(x, c_kappa_x)       
    #         #local_affinity[i] = cosine(x, c_kappa_x) 
    #     sim_lcent = sim - (local_affinity ** gamma)
    #     return 1 - sim_lcent
    #===========================================================================
    
    def localized_centering(self, kappa:int=20, gamma:float=1, 
                        distance_metric=Distance.cosine, test_set_mask=None):
        """Perform localized centering.
        
        Returns a distance matrix (not centered vectors!)
        Default parameters: kappa=20
        """
        # TODO CHECK CORRECTNESS!!
        if test_set_mask is None:
            test_set_mask = np.zeros(self.vectors.shape[0], np.bool)
            
        #if kappa == None:
        #    kappa = 20
        #    print("No 'kappa' defined for localized centering. \
        #        Using kappa=20 for 20 nearest neighbors.")
         
        if distance_metric == Distance.cosine:   
            # Rescale vectors to unit length
            v = self.vectors / np.sqrt((self.vectors ** 2).sum(-1))[..., np.newaxis]
            # for unit vectors it holds inner() == cosine()
            sim = 1 - cosine_distance(v)
        elif distance_metric == Distance.euclidean:
            v = self.vectors # no scaling here...
            sim = 1 / ( 1 + euclidean_distance(v))
        else:
            raise ValueError("Localized centering currently only supports "
                             "cosine or euclidean distances.")
        n = sim.shape[0]
        local_affinity = np.zeros(n)
        for i in range(n):
            x = v[i]
            sim_i = sim[i, :]
            # set similarity to test examples to zero to exclude them from fit
            sim_i[test_set_mask] = 0 
            #TODO randomization
            nn = np.argsort(sim_i)[::-1][1 : kappa+1]
            c_kappa_x = np.mean(v[nn], 0)
            if distance_metric == Distance.cosine:
                # c_kappa_x has no unit length in general
                local_affinity[i] = np.inner(x, c_kappa_x)       
                #local_affinity[i] = cosine(x, c_kappa_x) 
            elif distance_metric == Distance.euclidean:
                local_affinity[i] = 1 / (1 + np.linalg.norm(x-c_kappa_x))
            else:
                raise ValueError("Localized centering currently only supports "
                                 "cosine or euclidean distances.")
        sim_lcent = sim - (local_affinity ** gamma)
        return 1 - sim_lcent

if __name__ == '__main__':
    vectors = np.arange(12).reshape(3,4)
    c = Centering(vectors)
    print("Centering: ............. {}".format(c.centering()))
    print("Weighted centering: .... {}".format(c.weighted_centering(0.4)))
    print("Localized centering: ... {}".format(c.localized_centering(2)))