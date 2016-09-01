.. _installation:

============
Installation
============


Most of the instructions below assume you are running a Linux system. 
It might be possible to install the Hub Toolbox on Mac or Windows systems.
We cannot, however, give any guidance for these cases at this point.


Prerequisites
=============

Python
------

The Hub Toolbox currently requires Python 3.4 or higher. You can check this 
on your system with:

.. code-block:: bash

	python3 --version

If Python3 is missing, or its version is lower than 3.4, please install it 
via the package manager of your operating system (e.g. ``apt`` in 
Debian/Ubuntu or ``dnf`` in Fedora).

You might also consider using the `Anaconda environment 
<https://www.continuum.io/downloads#linux>`_ for easy Python environment 
and package handling.

numpy/scipy/scikit-learn
------------------------

The Hub Toolbox heavily relies on numpy and requires scipy and scikit-learn
for some functions.
Please install these packages via your operating system's package manager
(e.g. ``sudo apt install python3-numpy python3-scipy python3-sklearn``) or
use Anaconda: ``conda install numpy scipy scikit-learn``.
We do not recommend installation via ``pip`` since this may lead to suboptimal
performance unless configured properly.


Stable Hub Toolbox release
==========================

Stable releases of the Hub Toolbox are added to 
`PyPI <https://pypi.python.org/pypi/hub-toolbox>`_ .
To install the latest stable release, simply use `pip` 
(you may need to install it first via your operating system's package manager,
e.g. ``sudo apt install python3-pip``).

.. code-block:: bash

	pip3 install hub-toolbox

Alternatively, you may download the `latest release from GitHub 
<https://github.com/OFAI/hub-toolbox-python3/releases/latest>`_ and follow
the instructions of a development installation (from source) below,
omitting the ``git clone`` step.


.. _hubtoolbox-development-install:

Installation from source
========================

For a bleeding edge version of the Hub Toolbox, you can install it from
the latest sources:  
On the console, change to the directory, under which the Hub Toolbox should
be installed. Then obtain a copy of the latest sources from GitHub:

.. code-block:: bash

  git clone https://github.com/OFAI/hub-toolbox-python3.git

They will be cloned to a subdirectory called ``hub-toolbox-python3``. 
The Hub Toolbox must then be built and installed with

.. code-block:: bash

	cd hub-toolbox-python3
	python3 setup.py build
	sudo python3 setup.py install
	
The Hub Toolbox is now available system wide. Optionally, you can now run
a test suite by

.. code-block:: bash

	sudo python3 setup.py test
	
If this prints an ``OK`` message, you are ready to go. Note, that some 
skipped tests are fine.
