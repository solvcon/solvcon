#!/bin/sh
conda install \
  six setuptools sphinx ipython jupyter \
  cython numpy netcdf4 nose paramiko boto
conda install -c https://conda.anaconda.org/yungyuc \
  gmsh graphviz scotch
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
pip install -U sphinxcontrib-issuetracker
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
