#!/bin/sh
conda install -c https://conda.binstar.org/yungyuc/channel/solvcon \
  setuptools mercurial conda-build \
  scons cython numpy netcdf4 scotch nose paramiko boto \
  gmsh vtk sphinx graphviz
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
easy_install -UZ sphinxcontrib-issuetracker
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
