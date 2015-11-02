"""
Performs a quick hubness analysis with some of the functions provided in this 
toolbox.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-15

"""

import numpy as np
from hub_toolbox.Hubness import Hubness
from hub_toolbox.KnnClassification import KnnClassification
from hub_toolbox.GoodmanKruskal import GoodmanKruskal
from hub_toolbox.IntrinsicDim import IntrinsicDim
from hub_toolbox.MutualProximity import MutualProximity, Distribution
from hub_toolbox.LocalScaling import LocalScaling
from hub_toolbox.SharedNN import SharedNN
from hub_toolbox.Centering import Centering
from hub_toolbox import Distances as htd


class HubnessAnalysis():
    """
    The main hubness analysis class.
    
    Usage:
        hubness_analysis() - Loads the example data set and performs the
        analysis

    hubness_analysis(D, classes, vectors) - Uses the distance matrix D (NxN)
        together with an optional class labels vector (classes) and the
        original (optional) data vectors (vectors) to perform a full hubness
        analysis
    """
    
    def __init__(self, D = None, classes = None, vectors = None):
        """The constructor for a quick hubness analysis."""        
        
        self.haveClasses, self.haveVectors = False, False
        if D is None:
            self.D, self.classes, self.vectors = self.load_dexter()
            self.haveClasses, self.haveVectors = True, True
        else:
            self.D = np.copy(D)
           
            self.classes = np.copy(classes)
            self.vectors = np.copy(vectors)
            if classes is not None:
                self.haveClasses = True
            if vectors is not None:
                self.haveVectors = True
        self.n = len(self.D)
                
    def analyse_hubness(self, origData=True, mp=True, mp_gauss=False, \
                        mp_gaussi=True, mp_gammai=True, ls=True, snn=True, \
                        cent=True, wcent=True, wcent_g=0.4, \
                        lcent=True, lcent_k=40, lcent_g=1.4):
        """Analyse hubness in original data and rescaled distances.
        
        Use boolean parameters to choose which analyses to perform.        
        Rescale algorithms: Mutual Proximity (empiric, gaussian, independent 
        gaussian, independent gamma), Local Scaling, Shared Nearest Neighbors,
        Centering, Weighted Centering, Localized Centering"""
        
        print()
        print("Hubness Analysis")
            
        if origData:
            # Hubness in original data
            hubness = Hubness(self.D) 
            # Get hubness and n-occurence (slice omits elem 1, i.e. kNN)
            Sn5, Nk5 = hubness.calculate_hubness()[::2]
            self.print_results('ORIGINAL DATA', self.D, Sn5, Nk5, True)
        if mp or mp_gaussi or mp_gammai or mp_gauss:
            mut_prox = MutualProximity(self.D)
            if mp:  
                # Hubness in empiric mutual proximity distance space
                Dn = mut_prox.calculate_mutual_proximity(Distribution.empiric)
                hubness = Hubness(Dn)
                Sn5, Nk5 = hubness.calculate_hubness()[::2]
                self.print_results('MUTUAL PROXIMITY (Empiric/Slow)', \
                                   Dn, Sn5, Nk5)
            if mp_gauss:    
                # Hubness in mutual proximity distance space, Gaussian model
                Dn = mut_prox.calculate_mutual_proximity(Distribution.gauss)
                hubness = Hubness(Dn)
                Sn5, Nk5 = hubness.calculate_hubness()[::2]
                self.print_results('MUTUAL PROXIMITY (Gaussian)', Dn, Sn5, Nk5)
            if mp_gaussi:
                # Hubness in mutual proximity distance space, independent Gaussians
                Dn = mut_prox.calculate_mutual_proximity(Distribution.gaussi)
                hubness = Hubness(Dn)
                Sn5, Nk5 = hubness.calculate_hubness()[::2]
                self.print_results('MUTUAL PROXIMITY (Independent Gaussians)', \
                                   Dn, Sn5, Nk5)
            if mp_gammai:
                # Hubness in mutual proximity distance space, indep. Gamma distr.
                Dn = mut_prox.calculate_mutual_proximity(Distribution.gammai)
                hubness = Hubness(Dn)
                Sn5, Nk5 = hubness.calculate_hubness()[::2]
                self.print_results('MUTUAL PROXIMITY (Independent Gamma)', \
                                   Dn, Sn5, Nk5)
        if ls:
            # Hubness in local scaling distance space
            ls = LocalScaling(self.D, 10, 'original')
            Dn = ls.perform_local_scaling()
            hubness = Hubness(Dn)
            Sn5, Nk5 = hubness.calculate_hubness()[::2]
            self.print_results('LOCAL SCALING (Original, k=10)', Dn, Sn5, Nk5)
        if snn:
            # Hubness in shared nearest neighbors space
            snn = SharedNN(self.D, 10)
            Dn = snn.perform_snn()
            hubness = Hubness(Dn)
            Sn5, Nk5 = hubness.calculate_hubness()[::2]
            self.print_results('SHARED NEAREST NEIGHBORS (k=10)', Dn, Sn5, Nk5)
        if cent or wcent or lcent:
            if not self.vectors:
                print("Centering is currently only supported for vector data.")
            else:
                cent = Centering(self.vectors)
                if cent:
                    # Hubness after centering
                    D_cent = htd.cosine_distance(cent.centering())
                    hubness = Hubness(D_cent)
                    Sn5, Nk5 = hubness.calculate_hubness()[::2]
                    self.print_results('CENTERING', D_cent, Sn5, Nk5)
                if wcent:        
                    # Hubness after weighted centering
                    D_wcent = htd.cosine_distance(cent.weighted_centering(wcent_g))
                    hubness = Hubness(D_wcent)
                    Sn5, Nk5 = hubness.calculate_hubness()[::2]
                    self.print_results('WEIGHTED CENTERING (gamma={})'.format(\
                                        wcent_g), D_wcent, Sn5, Nk5)
                if lcent:
                    # Hubness after localized centering
                    D_lcent = 1 - cent.localized_centering(kappa=lcent_k, \
                                                           gamma=lcent_g)
                    hubness = Hubness(D_lcent)
                    Sn5, Nk5 = hubness.calculate_hubness()[::2]
                    self.print_results(\
                        'LOCALIZED CENTERING (k={}, gamma={})'.format(\
                        lcent_k, lcent_g), D_lcent, Sn5, Nk5)
    
    def print_results(self, heading : str, distances, Sn5 : float, Nk5 : float, 
                      calc_intrinsic_dimensionality : bool = False):
        """Print the results of a hubness analysis."""      
        
        print()
        print(heading + ':')
        print('data set hubness (S^n=5)                 : {:.3}'.format(Sn5))
        print('% of anti-hubs at k=5                    : {:.4}%'.format(\
            100 * sum(Nk5==0)/self.n))
        print('% of k=5-NN lists the largest hub occurs : {:.4}%'.format(\
            100 * max(Nk5)/self.n))
        if self.haveClasses:
            k = 5
            knn = KnnClassification(distances, self.classes, k)
            acc = knn.perform_knn_classification()[0]
            print('k=5-NN classification accuracy           : {:.4}%'.format(\
                    100*float(acc[0])))
                
            gk = GoodmanKruskal(distances, self.classes) 
            print('Goodman-Kruskal index (higher=better)    : {:.3}'.format(\
                gk.calculate_goodman_kruskal_index()))
        else:
            print('k=5-NN classification accuracy           : No classes given')
            print('Goodman-Kruskal index (higher=better)    : No classes given')
        
        if calc_intrinsic_dimensionality:
            if self.haveVectors:
                print('original dimensionality                  : {}'.format(\
                    np.size(self.vectors, 1)))
                idim = IntrinsicDim(self.vectors)
                print('intrinsic dimensionality estimate        : {}'.format(\
                    round(idim.calculate_intrinsic_dimensionality())))
            else:
                print('original dimensionality                  : No vectors given')
                print('intrinsic dimensionality estimate        : No vectors given')
        
    def load_dexter(self):
        """Load the example data set (dexter)."""
        
        print('\nNO PARAMETERS GIVEN! Loading & evaluating DEXTER data set.\n');
        print('DEXTER is a text classification problem in a bag-of-word');
        print('representation. This is a two-class classification problem');
        print('with sparse continuous input variables.');
        print('This dataset is one of five datasets of the NIPS 2003 feature');
        print('selection challenge.\n');
        print('http://archive.ics.uci.edu/ml/datasets/Dexter\n');
        
        import os
    
        n = 300
        dim = 20000
        
        # Read class labels
        classes_file = os.path.dirname(os.path.realpath(__file__)) +\
            '/example_datasets/dexter_train.labels'
        classes = np.loadtxt(classes_file)  

        # Read data
        vectors = np.zeros( (n, dim) )
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
        D = htd.cosine_distance(vectors)
        return D, classes, vectors
                
if __name__=="__main__":
    hub = HubnessAnalysis()
    hub.analyse_hubness()
    
    