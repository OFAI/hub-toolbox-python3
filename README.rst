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

The Hub Toolbox is a collection of scripts for the analysis and 
reduction of hubness in high-dimensional data. 

It allows you to

- analyze, whether your datasets show hubness
- reduce hubness via a variety of different techniques 
  (including scaling and centering approaches)
  and obtain secondary distances for downstream analysis inside or 
  outside the Hub Toolbox
- perform evaluation tasks with both internal and external measures
  (e.g. Goodman-Kruskal index and k-NN classification) 
	
Installation
------------

Make sure you have a working Python3 environment (at least 3.4) with
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


Citation
--------

If you use the Hub Toolbox in your scientific publication, please cite:

.. code-block:: text

	@article{feldbauer2016a,
			 title={Centering Versus Scaling for Hubness Reduction},
			 author={Feldbauer, Roman and Flexer, Arthur},
			 book={Artificial Neural Networks and Machine Learning - ICANN 2016},
			 year={2016},
			 url={http://www.ofai.at/cgi-bin/tr-online?number+2016-05}
	}

or

.. code-block:: text

	@article{schnitzer2012local,
			 title={Local and global scaling reduce hubs in space},
			 author={Schnitzer, Dominik and Flexer, Arthur and 
			 		 Schedl, Markus and Widmer, Gerhard},
			 journal={Journal of Machine Learning Research},
			 volume={13},
			 pages={2871--2902},
			 year={2012}
	}
	
License
-------
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

Acknowledgements
----------------
PyVmMonitor is being used to support the development of this free open source 
software package. For more information go to http://www.pyvmmonitor.com
	
