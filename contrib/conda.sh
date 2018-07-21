#!/bin/bash
conda install -y \
  python=3.6 \
  cmake setuptools pip sphinx ipython jupyter \
  cython numpy hdf4 netcdf4 nose pytest paramiko boto graphviz
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
conda install -y -c https://conda.anaconda.org/yungyuc \
  gmsh scotch
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
pip install pybind11 pythreejs
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
