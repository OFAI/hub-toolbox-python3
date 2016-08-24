Which Hub Toolbox to choose
===========================

The Hub Toolbox comes in two flavors. 

If in doubt, use the Hub Toolbox for Python.

For a more detailed description, see below.

hub-toolbox-matlab
--------------------

The Hub Toolbox was originally developed for Matlab/Octave. 
We still provide this flavor, however, development is limited to bugfixing.
No new functionality will be added (you may, however, get in contact with us 
at GitHub, if you wish to contribute).
The Hub Toolbox for Matlab supports:

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

The Hub Toolbox for Python3 was initially ported from the Matlab code. 
Development now focuses on this flavor. It is thus continuously being extended 
for new functionality and is tested and documented thoroughly. 
The Hub Toolbox for Python3 offers all the functionality the Matlab 
flavor offers, plus:

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
assume you are using this flavor.
