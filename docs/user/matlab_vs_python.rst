Which Hub Toolbox to choose
===========================

The Hub Toolbox is available as Python and Matlab scripts. 
If in doubt, use the Hub Toolbox for Python. See below
for a more detailed description.

hub-toolbox-matlab
--------------------

The Hub Toolbox was originally developed for Matlab/Octave. 
We still provide these scripts, however, development is limited to bugfixing.
No new functionality will be added.
The `Hub Toolbox for Matlab <https://github.com/OFAI/hub-toolbox-matlab>`_ 
supports:

- hubness analysis

- hubness reduction

  - Mutual Proximity
  - Local Scaling
  - Shared Nearest Neighbors
- evaluation

  - k-NN classification
  - Goodman-Kruskal index

for distance matrices.

hub-toolbox-python3
-------------------

The `Hub Toolbox for Python3 <https://github.com/OFAI/hub-toolbox-python3>`_ 
was initially ported from the Matlab code. 
Development now focuses on these scripts. It is thus continuously being extended 
for new functionality and is tested and documented thoroughly. 
The Hub Toolbox for Python3 offers all the functionality the Matlab 
scripts offer, plus:

- additional hubness reduction methods

  - centering
  - DisSim
- using similarity matrices instead of distance matrices
- support for sparse matrices (some modules)
- support for parallel processing (some modules)
- performance improvements (some modules)
- unit tests
- this documentation
 
We recommend using hub-toolbox-python3 for all users. This documentation will 
assume you are using these scripts.
