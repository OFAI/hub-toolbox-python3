#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2018, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
from inspect import signature
import numpy as np
from hub_toolbox.hubness import hubness
from hub_toolbox.knn_classification import score
from hub_toolbox.goodman_kruskal import goodman_kruskal_index
from hub_toolbox.intrinsic_dimension import intrinsic_dimension
from hub_toolbox.global_scaling import mutual_proximity_empiric, \
    mutual_proximity_gammai, mutual_proximity_gauss, mutual_proximity_gaussi
from hub_toolbox.local_scaling import nicdm, local_scaling
from hub_toolbox.shared_neighbors import shared_nearest_neighbors
from hub_toolbox.centering import centering, weighted_centering, \
    localized_centering, dis_sim_global, dis_sim_local
from hub_toolbox.distances import cosine_distance
from hub_toolbox import io

__all__ = ['HubnessAnalysis']

CITATION = \
"""
Feldbauer, R., Flexer, A.: Centering Versus Scaling for Hubness Reduction.
In: Villa, E.P.A., Masulli, P., Pons Rivero, J.A. (eds.) ICANN 2016, part I.
LNCS, vol. 9886, pp. 175–183. Springer International Publishing, Cham (2016).
(tech report available at http://www.ofai.at/cgi-bin/tr-online?number+2016-05)

or

Schnitzer, D., Flexer, A., Schedl, M., Widmer, G.: Local and global scaling
reduce hubs in space. J. Mach. Learn. Res. 13(1), 2871–2902 (2012).
(full paper available at http://www.jmlr.org/papers/v13/schnitzer12a.html)
"""

def _primary_distance(D:np.ndarray, metric):
    """Return `D`, identical. (Dummy function.)"""
    return D

# New types of hubness reduction methods must be added here
SEC_DIST = {'mp' : mutual_proximity_empiric,
            'mp_gauss': mutual_proximity_gauss,
            'mp_gaussi' : mutual_proximity_gaussi,
            'mp_gammai' : mutual_proximity_gammai,
            'ls' : local_scaling,
            'nicdm' : nicdm,
            'snn' : shared_nearest_neighbors,
            'cent' : centering,
            'wcent' : weighted_centering,
            'lcent' : localized_centering,
            'dsg' : dis_sim_global,
            'dsl' : dis_sim_local,
            'orig' : _primary_distance # a dummy function
           }

class HubnessAnalysis():
    """The main hubness analysis class.

    For more detailed analyses (optimizing parameters, using similarity data,
    etc.) please use the individual modules.

    Examples
    --------
    >>> from hub_toolbox.HubnessAnalysis import HubnessAnalysis
    >>> hub = HubnessAnalysis()
    >>> hub.analyse_hubness()

    >>> hub = HubnessAnalysis(D, classes, vectors)
    >>> hub.analyse_hubness()

    Notes
    -----
    The first example loads the example data set and performs a quick 
    hubness analysis with some of the functions provided in this toolbox.

    For the second example you must provide a distance matrix `D` (NxN)
    together with an optional class labels vector (`classes`) and the
    original (optional) data vectors (`vectors`) to perform a full hubness
    analysis.

    See also
    --------
    analyse_hubness : additional parameters (e.g. k-occurence, k-NN)
    """

    def __init__(self, D:np.ndarray=None, classes:np.ndarray=None,
                 vectors:np.ndarray=None, metric:str='distance'):
        """Initialize a quick hubness analysis.

        Parameters
        ----------
        D : ndarray, optional (default: None)
            The n x n symmetric distance (similarity) matrix.
            Default: load example dataset (dexter).

        classes : ndarray, optional (default: None)
            The 1 x n class labels. Required for k-NN, GK.

        vectors : ndarray, optional (default: None)
            The m x n vector data. Required for IntrDim estimation.

        metric : {'distance', 'similarity'}
            Define whether `D` is a distance or similarity matrix.
        """

        self.has_class_data, self.has_vector_data = False, False
        if D is None:
            print('\n'
                  'NO PARAMETERS GIVEN! Loading & evaluating DEXTER data set.'
                  '\n'
                  'DEXTER is a text classification problem in a bag-of-word \n'
                  'representation. This is a two-class classification problem\n'
                  'with sparse continuous input variables. \n'
                  'This dataset is one of five datasets of the NIPS 2003\n'
                  'feature selection challenge.\n'
                  'http://archive.ics.uci.edu/ml/datasets/Dexter\n')
            self.D, self.classes, self.vectors = io.load_dexter()
            self.has_class_data, self.has_vector_data = True, True
            self.metric = 'distance'
        else:
            # copy data and ensure correct type (not int16 etc.)
            self.D = np.copy(D).astype(np.float64)
            if classes is None:
                self.classes = None
            else:
                self.classes = np.copy(classes).astype(np.float64)
                self.has_class_data = True
            if vectors is None:
                self.vectors = None
            else:
                self.vectors = np.copy(vectors).astype(np.float64)
                self.has_vector_data = True
            self.metric = metric
        self.n = len(self.D)
        self.experiments = []

    @property
    def _header(self):
        return {'mp' : "MUTUAL PROXIMITY (Empiric)",
                'mp_gauss': "MUTUAL PROXIMITY (Gaussian)",
                'mp_gaussi' : "MUTUAL PROXIMITY (Independent Gaussians)",
                'mp_gammai' : "MUTUAL PROXIMITY (Independent Gamma)",
                'ls' : "LOCAL SCALING (original)",
                'nicdm' : "LOCAL SCALING (NICDM)",
                'snn' : "SHARED NEAREST NEIGHBORS",
                'cent' : "CENTERING",
                'wcent' : "WEIGHTED CENTERING",
                'lcent' : "LOCALIZED CENTERING",
                'dsg' : "DISSIM GLOBAL",
                'dsl' : "DISSIM LOCAL",
                'orig' : "ORIGINAL DATA"}

    def _calc_intrinsic_dim(self):
        """Calculate intrinsic dimension estimate."""
        self.intrinsic_dim = intrinsic_dimension(X=self.vectors)
        return self

    def analyze_hubness(self, experiments="orig,mp,mp_gaussi,nicdm,cent,dsg",
                        hubness_k=(5, 10), knn_k=(1, 5, 20),
                        print_results=True, verbose:int=0):
        """Analyse hubness in original data and rescaled distances.

        Parameters
        ----------
        experiments : str, optional
            Define which experiments to perform. Please provide a string of
            comma separated values chosen from the following options:

            - "orig" : Original, primary distances
            - "mp" : Mutual Proximity (empiric)
            - "mp_gauss" : Mutual Proximity (Gaussians)
            - "mp_gaussi" : Mutual Proximity (independent Gaussians)
            - "mp_gammai" ... Mutual Proximity (independent Gamma)
            - "ls" : Local Scaling (using k-th neighbor)
            - "nicdm" : Local Scaling variant NICDM (average of k neighbors)
            - "snn" : Shared Nearest Neighbors
            - "cent" : Centering
            - "wcent" : Weighted Centering
            - "lcent" : Localized Centering
            - "dsg" : DisSim Global
            - "dsl" : DisSim Local

        hubness_k : tuple, optional (default: (5, 10))
            Hubness parameter (skewness of `k`-occurence)

        knn_k : tuple, optional (default: (1, 5, 20))
            `k`-NN classification parameter

        print_results : bool, optional (default: True)
            Define whether to print hubness analysis report to stdout

        verbose : int, optional (default: 0)
            Increasing output verbosity

        Returns
        -------
        self : optionally prints results to stdout
        """
        experiments = experiments.split(',')
        if self.vectors is None:
            self.intrinsic_dim = None
        else:
            self._calc_intrinsic_dim()
        for i, exp_type in enumerate(experiments):
            if verbose:
                print("Experiment {}/{} ({})".
                      format(i+1, len(experiments), exp_type), end="\r")
            experiment = HubnessExperiment(D=self.D,
                secondary_distance_type=exp_type, metric=self.metric,
                classes=self.classes, vectors=self.vectors)
            if self.D is not None:
                experiment._calc_secondary_distance()
                for k in hubness_k:
                    experiment._calc_hubness(k=k)
            if self.classes is not None:
                for k in knn_k:
                    experiment._calc_knn_accuracy(k=k)
                experiment._calc_gk_index()
            self.experiments.append(experiment)
            if print_results:
                self.print_analysis_report(experiment, report_nr=i)
        if print_results:
            print("------------------------------------------------------------")
            print("Thanks for using the HUB-TOOLBOX!")
            print("If you use this software in a research project, please cite:")
            print("\n", CITATION)
            print("Please also consider citing the references to the \n"
                  "individual modules/hubness functions that you use.")
        return self

    def print_analysis_report(self, experiment=None, report_nr:int=0):
        """Print a report of the performed hubness analysis.

        Parameters
        ----------
        experiment : HubnessExperiment, optional (default: None)
            If given, report only this `experiment`. Otherwise, report all
            experiments of this analysis.

        report_nr : int, optional (default: 0)
            Method only prints headline for first report

        Returns
        -------
        None : Output is printed to stdout
        """
        if experiment is not None:
            experiments = [experiment]
        else:
            experiments = self.experiments
        if report_nr == 0:
            print("\n"
                  "================\n"
                  "Hubness Analysis\n"
                  "================\n")
        for experiment in experiments:
            print(self._header[experiment.secondary_distance_type] + ':')
            # Print used parameters (which are the default parameters)
            sig = signature(SEC_DIST[experiment.secondary_distance_type])
            for p in ['k', 'kappa', 'gamma']:
                try:
                    print("parameter {} = {} (for optimization use the "
                          "individual modules of the HUB-TOOLBOX)".
                          format(p, sig.parameters[p].default))
                except KeyError:
                    pass # function does not use this parameter
            try: # to print hubness results, if available
                for k in sorted(experiment.hubness.keys()):
                    print('data set hubness (S^k={:2})                : {:.3}'.
                          format(k, experiment.hubness[k]))
                    print('% of anti-hubs at k={:2}                   : {:.4}%'.
                          format(k, experiment.anti_hubs[k]))
                    print('% of k={:2}-NN lists the largest hub occurs: {:.4}%'.
                          format(k, experiment.max_hub_k_occurence[k]))
            except KeyError:
                print('data set hubness (S^k={:2})                : '
                      'No k given')
            try: # to print k-NN results, if available
                for k in sorted(experiment.knn_accuracy.keys()):
                    print('k={:2}-NN classification accuracy          : {:.4}%'.
                          format(k, 100.*float(experiment.knn_accuracy[k])))
            except KeyError:
                print('k=5-NN classification accuracy           : '
                      'No classes given')
            # print Goodman-Kruskal result, if available
            if experiment.gk_index is None:
                print('Goodman-Kruskal index (higher=better)    : '
                      'No classes given/Not calculated')
            else:
                print('Goodman-Kruskal index (higher=better)    : {:.3}'.
                      format(experiment.gk_index))
            # Embedding dimension
            if self.vectors is None:
                print('embedding dimensionality                 : '
                      'No vectors given')
            else:
                print('embedding dimensionality                 : {}'.
                      format(experiment.embedding_dim))
            # Intrinsic dimension estimate, if available
            if self.intrinsic_dim is None:
                print('intrinsic dimensionality estimate        : '
                      'No vectors given')
            else:
                print('intrinsic dimensionality estimate        : {}'.
                      format(round(self.intrinsic_dim)))
            print()
        return

class HubnessExperiment():
    """Perform a single hubness experiment"""

    def __init__(self, D:np.ndarray, secondary_distance_type:str,
                 metric:str='distance', classes:np.ndarray=None,
                 vectors:np.ndarray=None):
        """Initialize a hubness experiment"""

        io.check_distance_matrix_shape(D)
        io.check_valid_metric_parameter(metric)
        if secondary_distance_type not in SEC_DIST.keys():
            raise ValueError("Requested secondary distance type unknown.")
        if classes is not None:
            io.check_distance_matrix_shape_fits_labels(D, classes)
        if vectors is None:
            self.embedding_dim = None
        else: # got vectors
            io.check_distance_matrix_shape_fits_vectors(D, vectors)
            self.embedding_dim = vectors.shape[1]
        self.original_distance = D
        self.secondary_distance_type = secondary_distance_type
        self.classes = classes
        self.vectors = vectors
        self.metric = metric
        self.n = D.shape[0]
        # Obtained later through functions:
        self.secondary_distance = None
        self.hubness = dict()
        self.anti_hubs = dict()
        self.max_hub_k_occurence = dict()
        self.knn_accuracy = dict()
        self.gk_index = None

    def _calc_secondary_distance(self):
        """Calculate secondary distances (e.g. Mutual Proximity)"""
        sec_dist_fun = SEC_DIST[self.secondary_distance_type]
        try:
            self.secondary_distance = sec_dist_fun(
                D=self.original_distance, metric=self.metric)
        except TypeError: # centering has no keyword 'D='
            if self.secondary_distance_type in ['cent', 'wcent']:
                self.secondary_distance = \
                    cosine_distance(sec_dist_fun(X=self.vectors))
            elif self.secondary_distance_type in ['lcent']:
                self.secondary_distance = 1. - sec_dist_fun(X=self.vectors)
            elif self.secondary_distance_type in ['dsg', 'dsl']:
                self.secondary_distance = sec_dist_fun(X=self.vectors)
            else:
                raise ValueError("Erroneous secondary distance type: {}".
                                 format(self.secondary_distance_type))
        return self

    def _calc_hubness(self, k:int=5):
        """Calculate hubness (skewness of `k`-occurence).

        Also calculate percentage of anti hubs (`k`-occurence == 0) and
        percentage of k-NN lists the largest hub occurs in.
        """
        S_k, _, N_k = hubness(D=self.secondary_distance,
                              metric=self.metric, k=k)
        self.hubness[k] = S_k
        self.anti_hubs[k] = 100 * (N_k == 0).sum() / self.n
        self.max_hub_k_occurence[k] = 100 * N_k.max() / self.n
        return self

    def _calc_knn_accuracy(self, k:int=5):
        """Calculate `k`-NN accuracy."""
        acc, _, _ = score(D=self.secondary_distance, target=self.classes,
                          k=k, metric=self.metric)
        self.knn_accuracy[k] = acc
        return self

    def _calc_gk_index(self):
        """Calculate Goodman-Kruskal's gamma."""
        self.gk_index = goodman_kruskal_index(D=self.secondary_distance,
                                              classes=self.classes,
                                              metric=self.metric)
        return self

if __name__ == "__main__":
    hub = HubnessAnalysis()
    hub.analyze_hubness()
