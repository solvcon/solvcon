#!/bin/bash
conda install -y \
  six setuptools pip sphinx ipython jupyter \
  cython numpy netcdf4 nose paramiko boto graphviz
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
conda install -y -c https://conda.anaconda.org/yungyuc \
  gmsh scotch
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
pip install pybind11 pythreejs sphinxcontrib-issuetracker
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
