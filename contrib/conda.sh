#!/bin/bash
conda install -y \
  cmake six setuptools pip sphinx ipython jupyter \
  cython numpy netcdf4 nose paramiko boto graphviz
# issue #178
conda install -y hdf4=4.2.12
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
conda install -y -c https://conda.anaconda.org/yungyuc \
  gmsh scotch
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
pip install pybind11 pythreejs sphinxcontrib-issuetracker
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
