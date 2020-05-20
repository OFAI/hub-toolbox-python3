.. image:: https://badge.fury.io/py/hub-toolbox.svg
    :target: https://badge.fury.io/py/hub-toolbox

.. image:: https://readthedocs.org/projects/hub-toolbox-python3/badge/?version=latest
	:target: http://hub-toolbox-python3.readthedocs.io/en/latest/?badge=latest
	:alt: Documentation Status

.. image:: https://travis-ci.org/OFAI/hub-toolbox-python3.svg?branch=master
    :target: https://travis-ci.org/OFAI/hub-toolbox-python3

.. image:: https://coveralls.io/repos/github/OFAI/hub-toolbox-python3/badge.svg?branch=master
	:target: https://coveralls.io/github/OFAI/hub-toolbox-python3?branch=master 

.. image:: https://img.shields.io/aur/license/yaourt.svg?maxAge=2592000   
	:target: https://github.com/OFAI/hub-toolbox-python3/blob/master/LICENSE.txt


HUB-TOOLBOX
===========

#-----------------------------------------------------------------------------------

Checkout our new project `scikit-hubness <https://github.com/VarIr/scikit-hubness>`_
which provides the functionality of the Hub-Toolbox while integrating nicely into
`scikit-learn` workflows.

Use `skhubness.neighbors` as a drop-in replacement for `sklearn.neighbors`.
It offers the same functionality and adds transparent support for hubness reduction,
approximate nearest neighbor search (HNSW, LSH), and approximate hubness reduction.

We strive to improve usability of hubness reduction with the development of
`scikit-hubness`, and we are very interested in
`user feedback <https://github.com/VarIr/scikit-hubness/issues>`_!

#-----------------------------------------------------------------------------------

The Hub Toolbox is a software suite for hubness analysis and
hubness reduction in high-dimensional data.

It allows to

- analyze, whether your datasets show hubness
- reduce hubness via a variety of different techniques 
  (including scaling and centering approaches)
  and obtain secondary distances for downstream analysis inside or 
  outside the Hub Toolbox
- perform evaluation tasks with both internal and external measures
  (e.g. Goodman-Kruskal index and k-NN classification)
- NEW IN 2.5:
  The ``approximate`` module provides approximate hubness reduction methods
  with linear complexity which allow to analyze large datasets.
- NEW IN 2.5:
  Measure hubness with the recently proposed Robin-Hood index
  for fast and reliable hubness estimation.
	
Installation
------------

Make sure you have a working Python3 environment (at least 3.6) with
numpy, scipy and scikit-learn packages. Use pip3 to install the latest 
stable version:

.. code-block:: bash

  pip3 install hub-toolbox

For more details and alternatives, please see the `Installation instructions
<http://hub-toolbox-python3.readthedocs.io/en/latest/user/installation.html>`_.

Documentation
-------------

Documentation is available online: 
http://hub-toolbox-python3.readthedocs.io/en/latest/index.html

Example
-------

To run a full hubness analysis on the example dataset (DEXTER) 
using some of the provided hubness reduction methods, 
simply run the following in a Python shell:

.. code-block:: python

	>>> from hub_toolbox.HubnessAnalysis import HubnessAnalysis
	>>> ana = HubnessAnalysis()
	>>> ana.analyze_hubness()
	
See how you can conduct the individual analysis steps:

.. code-block:: python

	import hub_toolbox
	
	# load the DEXTER example dataset
	D, labels, vectors = hub_toolbox.io.load_dexter()

	# calculate intrinsic dimension estimate
	d_mle = hub_toolbox.intrinsic_dimension.intrinsic_dimension(vector)
	
	# calculate hubness (here, skewness of 5-occurence)
	S_k, _, _ = hub_toolbox.hubness.hubness(D=D, k=5, metric='distance')

	# perform k-NN classification LOO-CV for two different values of k
	acc, _, _ = hub_toolbox.knn_classification.score(
                D=D, target=labels, k=[1,5], metric='distance')

	# calculate Goodman-Kruskal index
	gamma = hub_toolbox.goodman_kruskal.goodman_kruskal_index(
	    D=D, classes=labels, metric='distance')
	 	
	# Reduce hubness with Mutual Proximity (Empiric distance distribution)
	D_mp = hub_toolbox.global_scaling.mutual_proximity_empiric(
	    D=D, metric='distance')
		
	# Reduce hubness with Local Scaling variant NICDM
	D_nicdm = hub_toolbox.local_scaling.nicdm(D=D, k=10, metric='distance')
	
	# Check whether indices improve after hubness reduction
	S_k_mp, _, _ = hub_toolbox.hubness.hubness(D=D_mp, k=5, metric='distance')
	acc_mp, _, _ = hub_toolbox.knn_classification.score(
		D=D_mp, target=labels, k=[1,5], metric='distance')
	gamma_mp = hub_toolbox.goodman_kruskal.goodman_kruskal_index(
		D=D_mp, classes=labels, metric='distance')
		
	# Repeat the last steps for all secondary distances you calculated
	...

Check the `Tutorial
<http://hub-toolbox-python3.readthedocs.io/en/latest/user/tutorial.html>`_ 
for in-depth explanations of the same. 


Development
-----------

Development of the Hub Toolbox has finished. Check out its successor
`scikit-hubness <https://github.com/VarIr/scikit-hubness>`_ for fully
scikit-learn compatible hubness analysis and approximate neighbor search.

.. code-block:: text

	(c) 2011-2018, Dominik Schnitzer and Roman Feldbauer
	Austrian Research Institute for Artificial Intelligence (OFAI)
	Contact: <roman.feldbauer@ofai.at>

Citation
--------

If you use the Hub Toolbox in your scientific publication, please cite:

.. code-block:: text

	@InProceedings{Feldbauer2018b,
                   author        = {Roman Feldbauer and Maximilian Leodolter and Claudia Plant and Arthur Flexer},
                   title         = {Fast Approximate Hubness Reduction for Large High-Dimensional Data},
                   booktitle     = {2018 {IEEE} International Conference on Big Knowledge, {ICBK} 2018, Singapore, November 17-18, 2018},
                   year          = {2018},
                   editor        = {Xindong Wu and Yew{-}Soon Ong and Charu C. Aggarwal and Huanhuan Chen},
                   pages         = {358--367},
                   publisher     = {{IEEE} Computer Society},
                   bibsource     = {dblp computer science bibliography, https://dblp.org},
                   biburl        = {https://dblp.org/rec/conf/icbk/FeldbauerLPF18.bib},
                   doi           = {10.1109/ICBK.2018.00055},
                 }

Relevant literature:

2018: ``Fast approximate hubness reduction for large high-dimensional data``, available as
technical report at `<http://www.ofai.at/cgi-bin/tr-online?number+2018-02>`_.

2018: ``A comprehensive empirical comparison of hubness reduction in high-dimensional spaces``,
full paper available at https://doi.org/10.1007/s10115-018-1205-y

2016: ``Centering Versus Scaling for Hubness Reduction``, available as technical report
at `<http://www.ofai.at/cgi-bin/tr-online?number+2016-05>`_ .

2012: ``Local and Global Scaling Reduce Hubs in Space``, full paper available at
`<http://www.jmlr.org/papers/v13/schnitzer12a.html>`_ .

License
-------
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

Acknowledgements
----------------
PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com

