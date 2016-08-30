.. image:: https://readthedocs.org/projects/hub-toolbox-python3/badge/?version=latest
	:target: http://hub-toolbox-python3.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://travis-ci.org/OFAI/hub-toolbox-python3.svg?branch=master
    :target: https://travis-ci.org/OFAI/hub-toolbox-python3

.. image:: https://coveralls.io/repos/github/OFAI/hub-toolbox-python3/badge.svg?branch=master
	:target: https://coveralls.io/github/OFAI/hub-toolbox-python3?branch=master 

.. image:: https://img.shields.io/aur/license/yaourt.svg?maxAge=2592000   
	:target: https://github.com/OFAI/hub-toolbox-python3/blob/master/LICENSE.txt

A new version of this readme will replace this soon. 
	
.. code-block:: text

	-------------------------
	 HUB TOOLBOX VERSION 2.3 
	 August 22, 2016
	-------------------------
	
	This is the HUB TOOLBOX for Python3, licensed under the terms of the GNU GPLv3.
	(c) 2011-2016, Dominik Schnitzer and Roman Feldbauer
	Austrian Research Institute for Artificial Intelligence (OFAI)
	Contact: <roman.feldbauer@ofai.at>
	
	If you use the functions in your publication, please cite:
	
	@article{feldbauer2016a,
	  title={Centering Versus Scaling for Hubness Reduction},
	  author={Feldbauer, Roman and Flexer, Arthur},
	  book={Artificial Neural Networks and Machine Learning - ICANN 2016},
	  year={2016},
	  url={http://www.ofai.at/cgi-bin/tr-online?number+2016-05}
	}
	or
	@article{schnitzer2012local,
	  title={Local and global scaling reduce hubs in space},
	  author={Schnitzer, Dominik and Flexer, Arthur and Schedl, Markus and Widmer,
	    Gerhard},
	  journal={Journal of Machine Learning Research},
	  volume={13},
	  pages={2871--2902},
	  year={2012}
	}
	
	The full publication is available at:
	http://jmlr.org/papers/volume13/schnitzer12a/schnitzer12a.pdf
	
	
	The HUB TOOLBOX is a collection of hub/anti-hub analysis tools. To quickly
	try the various scaling functions on your distance matrices and evaluate their
	impact use the HubnessAnalysis module:
	
	>>> from hub_toolbox.HubnessAnalysis import HubnessAnalysis
	>>> analysis = HubnessAnalysis(D, classes, vectors)
	>>> analysis.analyze_hubness() 
	
	'D' is your (NxN) distance matrix, 'classes' is an optional vector with a
	class number per item in the rows of D. 'vectors' is the optional original data
	vectors. The function will output various hubness measurements, tries to remove
	hubs and evaluates the input data again.
	
	Internally the function uses the:
	  *  mutual_proximity_empiric(D),
	  *  mutual_proximity_gaussi(D),
	  *  nicdm(D, k=7),
	  *  centering(vectors)
	  *  dis_sim_global(vectors)
	functions to reduce hubness with different methods, and
	  *  hubness(D, k=[5, 10]),
	  *  score(D, classes, k=[1, 5, 20]), # k-NN classification
	  *  goodman_kruskal_index(D, classes),
	  *  intrinsic_dimension(vectors),
	to do the hubness analysis. Use the functions separately to do a more specific
	analysis of your own data.
	
	--------------------------------------
	 EXAMPLE WITH BUNDLED DEXTER DATA SET
	--------------------------------------
	
	If no parameter to HubnessAnalysis() is given, the DEXTER data set is loaded
	and evaluated. See example_datasets/ABOUT for more information about the data.
	
	$ python3 HubnessAnalysis.py 
	
	NO PARAMETERS GIVEN! Loading & evaluating DEXTER data set.
	
	DEXTER is a text classification problem in a bag-of-word
	representation. This is a two-class classification problem
	with sparse continuous input variables.
	This dataset is one of five datasets of the NIPS 2003 feature
	selection challenge.
	
	http://archive.ics.uci.edu/ml/datasets/Dexter
	
	
	================
	Hubness Analysis
	================
	
	ORIGINAL DATA:
	data set hubness (S^k= 5)                : 4.22
	% of anti-hubs at k= 5                   : 26.67%
	% of k= 5-NN lists the largest hub occurs: 23.67%
	data set hubness (S^k=10)                : 3.98
	% of anti-hubs at k=10                   : 17.67%
	% of k=10-NN lists the largest hub occurs: 50.0%
	k= 1-NN classification accuracy          : 80.33%
	k= 5-NN classification accuracy          : 80.33%
	k=20-NN classification accuracy          : 84.33%
	Goodman-Kruskal index (higher=better)    : 0.104
	embedding dimensionality                 : 20000
	intrinsic dimensionality estimate        : 161
	
	MUTUAL PROXIMITY (Empiric):
	data set hubness (S^k= 5)                : 0.643
	% of anti-hubs at k= 5                   : 3.0%
	% of k= 5-NN lists the largest hub occurs: 6.0%
	data set hubness (S^k=10)                : 0.721
	% of anti-hubs at k=10                   : 0.0%
	% of k=10-NN lists the largest hub occurs: 10.67%
	k= 1-NN classification accuracy          : 82.67%
	k= 5-NN classification accuracy          : 90.0%
	k=20-NN classification accuracy          : 88.33%
	Goodman-Kruskal index (higher=better)    : 0.132
	embedding dimensionality                 : 20000
	intrinsic dimensionality estimate        : 161
	
	MUTUAL PROXIMITY (Independent Gaussians):
	data set hubness (S^k= 5)                : 0.805
	% of anti-hubs at k= 5                   : 4.667%
	% of k= 5-NN lists the largest hub occurs: 5.667%
	data set hubness (S^k=10)                : 1.21
	% of anti-hubs at k=10                   : 0.0%
	% of k=10-NN lists the largest hub occurs: 12.67%
	k= 1-NN classification accuracy          : 83.67%
	k= 5-NN classification accuracy          : 89.0%
	k=20-NN classification accuracy          : 90.0%
	Goodman-Kruskal index (higher=better)    : 0.135
	embedding dimensionality                 : 20000
	intrinsic dimensionality estimate        : 161
	
	LOCAL SCALING (NICDM):
	parameter k = 7 (for optimization use the individual modules of the HUB-TOOLBOX)
	data set hubness (S^k= 5)                : 2.1
	% of anti-hubs at k= 5                   : 0.6667%
	% of k= 5-NN lists the largest hub occurs: 8.667%
	data set hubness (S^k=10)                : 1.74
	% of anti-hubs at k=10                   : 0.0%
	% of k=10-NN lists the largest hub occurs: 16.0%
	k= 1-NN classification accuracy          : 84.67%
	k= 5-NN classification accuracy          : 85.0%
	k=20-NN classification accuracy          : 85.0%
	Goodman-Kruskal index (higher=better)    : 0.118
	embedding dimensionality                 : 20000
	intrinsic dimensionality estimate        : 161
	
	CENTERING:
	data set hubness (S^k= 5)                : 1.62
	% of anti-hubs at k= 5                   : 6.667%
	% of k= 5-NN lists the largest hub occurs: 8.333%
	data set hubness (S^k=10)                : 1.38
	% of anti-hubs at k=10                   : 1.333%
	% of k=10-NN lists the largest hub occurs: 13.0%
	k= 1-NN classification accuracy          : 85.0%
	k= 5-NN classification accuracy          : 87.67%
	k=20-NN classification accuracy          : 89.33%
	Goodman-Kruskal index (higher=better)    : 0.19
	embedding dimensionality                 : 20000
	intrinsic dimensionality estimate        : 161
	
	DISSIM GLOBAL:
	data set hubness (S^k= 5)                : 1.87
	% of anti-hubs at k= 5                   : 6.333%
	% of k= 5-NN lists the largest hub occurs: 8.667%
	data set hubness (S^k=10)                : 1.62
	% of anti-hubs at k=10                   : 1.667%
	% of k=10-NN lists the largest hub occurs: 14.67%
	k= 1-NN classification accuracy          : 84.0%
	k= 5-NN classification accuracy          : 88.67%
	k=20-NN classification accuracy          : 88.67%
	Goodman-Kruskal index (higher=better)    : 0.189
	embedding dimensionality                 : 20000
	intrinsic dimensionality estimate        : 161
	
	------------------------------------------------------------
	Thanks for using the HUB-TOOLBOX!
	If you use this software in a research project, please cite:
	--- 
	Feldbauer, R., Flexer, A. (2016). Centering Versus Scaling for 
	Hubness Reduction. ICANN 2016, Part I, LNCS 9886, pp. 1–9 (preprint 
	available at http://www.ofai.at/cgi-bin/tr-online?number+2016-05).
	or
	Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). Local 
	and global scaling reduce hubs in space. The Journal of Machine 
	Learning Research, 13(1), 2871–2902.
	
	Please also consider citing the references to the 
	individual modules/hubness functions that you use.
	
	
	--------------
	 REQUIREMENTS
	--------------
	+ Python3
	+ NumPy
	+ SciPy
	+ scikit-learn
	The authors suggest using the Anaconda environment.
	
	-------
	LICENSE
	-------
	The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.
	
	----------------
	ACKNOWLEDGEMENTS
	----------------
	PyVmMonitor is being used to support the development of this free open source 
	software package. For more information go to http://www.pyvmmonitor.com
	
