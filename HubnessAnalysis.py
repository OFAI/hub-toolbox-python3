"""
Performs a quick hubness analysis with all the functions provided in this 
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
            #self.D, self.classes, self.vectors = self.load_small_data()
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
                
    def analyse_hubness(self):
        """Analyse hubness in original data and rescaled distances.
        
        Rescale algorithms: Mutual Proximity (empiric), 
        Local Scaling, Shared Nearest Neighbors"""
        
        print()
        print("Hubness Analysis")
            
        #"""    
        # Hubness in original data
        hubness = Hubness(self.D) 
        # Get hubness and n-occurence (slice omits elem 1, i.e. kNN)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('ORIGINAL DATA', self.D, Sn5, Nk5, True)
        #"""
        
        # Hubness in empiric mutual proximity distance space
        mut_prox = MutualProximity(self.D)
        Dn = mut_prox.calculate_mutual_proximity(Distribution.empiric)
        hubness = Hubness(Dn)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('MUTUAL PROXIMITY (Empiric/Slow)', Dn, Sn5, Nk5)
        """
        # Hubness in mutual proximity distance space, Gaussian model
        Dn = mut_prox.calculate_mutual_proximity(Distribution.gauss)
        hubness = Hubness(Dn)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('MUTUAL PROXIMITY (Gaussian)', Dn, Sn5, Nk5)
        """
        # Hubness in mutual proximity distance space, independent Gaussians
        Dn = mut_prox.calculate_mutual_proximity(Distribution.gaussi)
        hubness = Hubness(Dn)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('MUTUAL PROXIMITY (Independent Gaussians)', \
                           Dn, Sn5, Nk5)
        
        # Hubness in mutual proximity distance space, independent Gamma distr.
        Dn = mut_prox.calculate_mutual_proximity(Distribution.gammai)
        hubness = Hubness(Dn)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('MUTUAL PROXIMITY (Independent Gamma)', Dn, Sn5, Nk5)
        
        # Hubness in local scaling distance space
        ls = LocalScaling(self.D, 10, 'original')
        Dn = ls.perform_local_scaling()
        hubness = Hubness(Dn)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('LOCAL SCALING (Original, k=10)', Dn, Sn5, Nk5)
        
        # Hubness in shared nearest neighbors space
        snn = SharedNN(self.D, 10)
        Dn = snn.perform_snn()
        hubness = Hubness(Dn)
        Sn5, Nk5 = hubness.calculate_hubness()[::2]
        self.print_results('SHARED NEAREST NEIGHBORS (k=10)', Dn, Sn5, Nk5)
          
          
        
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
    
        n = 300
        dim = 20000
        
        # Read class labels
        classes = np.loadtxt('example_datasets/dexter_train.labels')  
        #classes = np.loadtxt('example_datasets/small.labels')

        # Read data
        vectors = np.zeros( (n, dim) )
        with open('example_datasets/dexter_train.data', mode='r') as fid:
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
        
    def load_small_data(self):
        """Load the example data set (dexter)."""
        
        print('\nNO PARAMETERS GIVEN! Loading & evaluating SMALL data set.\n');
        print('SMALL is artificial data, 6 points, 10 dim, sparse, 2 classes.');
        
       
        n = 6
        dim = 10
        
        # Read class labels
        classes = np.loadtxt('example_datasets/small.labels')

        # Read data
        vectors = np.zeros( (n, dim) )
        with open('example_datasets/small.data', mode='r') as fid:
        #with open('example_datasets/small2.data', mode='r') as fid:
        #small2.data: only -1 labels
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
        
def cosine_distance(x):
    """Calculate the cosine distance."""
    
    xn = np.sqrt(np.sum(x**2, 1))
    x = x / np.tile(xn[:, np.newaxis], np.size(x, 1))
    D = 1 - np.dot(x, x.T )
    #np.clip(D, 0, np.finfo(np.float64).max, out=D) # clip max set to MaxFloat
    D[D<0] = 0
    D = np.triu(D, 0) + np.triu(D, 0).T
    
    return D

if __name__=="__main__":
    hub = HubnessAnalysis()
    hub.analyse_hubness()
    
    