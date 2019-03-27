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
	D, labels, vectors = hub_toolbox.IO.load_dexter()

	# calculate intrinsic dimension estimate
	d_mle = hub_toolbox.IntrinsinDim.intrinsic_dimension(vectors)
	
	# calculate hubness (here, skewness of 5-occurence)
	S_k, _, _ = hub_toolbox.Hubness.hubness(D=D, k=5, metric='distance')
	
	# perform k-NN classification LOO-CV for two different values of k
	acc, _, _ = hub_toolbox.KnnClassification.score(
		D=D, target=labels, k=[1,5], metric='distance')

	# calculate Goodman-Kruskal index
	gamma = hub_toolbox.GoodmanKruskal.goodman_kruskal_index(
		D=D, classes=labels, metric='distance')
	 	
	# Reduce hubness with Mutual Proximity (Empiric distance distribution)
	D_mp = hub_toolbox.MutualProximity.mutual_proximity_empiric(
		D=D, metric='distance')
		
	# Reduce hubness with Local Scaling variant NICDM
	D_nicdm = hub_toolbox.LocalScaling.nicdm(D=D, k=10, metric='distance')
	
	# Check whether indices improve after hubness reduction
	S_k_mp, _, _ = hub_toolbox.Hubness.hubness(D=D_mp, k=5, metric='distance')
	acc_mp, _, _ = hub_toolbox.KnnClassification.score(
		D=D_mp, target=labels, k=[1,5], metric='distance')
	gamma_mp = hub_toolbox.GoodmanKruskal.goodman_kruskal_index(
		D=D_mp, classes=labels, metric='distance')
		
	# Repeat the last steps for all secondary distances you calculated
	...

Check the `Tutorial
<http://hub-toolbox-python3.readthedocs.io/en/latest/user/tutorial.html>`_ 
for in-depth explanations of the same. 


Development
-----------

The Hub Toolbox is a work in progress. Get in touch with us if you have
comments, would like to see an additional feature implemented, would like
to contribute code or have any other kind of issue. Please don't hesitate
to file an `issue <https://github.com/OFAI/hub-toolbox-python3/issues>`_ 
here on GitHub. 

.. code-block:: text

	(c) 2011-2018, Dominik Schnitzer and Roman Feldbauer
	Austrian Research Institute for Artificial Intelligence (OFAI)
	Contact: <roman.feldbauer@ofai.at>

Citation
--------

If you use the Hub Toolbox in your scientific publication, please cite:

.. code-block:: text

	@Inbook{Feldbauer2018b,
		author="Feldbauer, Roman
		and Leodolter, Maximilian
		and Plant, Claudia
		and Flexer, Arthur",
		editor="",
		title="Fast approximate hubness reduction for large high-dimensional data",
		bookTitle="IEEE International Conference on Big Knowledge, Singapore, 2018",
		year="2018",
		publisher="IEEE",
		address="",
		pages="",
		isbn="",
		doi="",
		url="",
		notes="(in press)"
		}

(We expect the proceedings to published by the IEEE in Dec 2018).

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

