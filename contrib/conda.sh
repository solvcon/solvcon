#!/bin/bash
conda install -y \
  python=3.8 \
  cmake \
  numpy hdf4 netcdf4 nose boto paramiko
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
conda install -y -c https://conda.anaconda.org/yungyuc scotch
lret=$?; if [[ $lret != 0 ]] ; then exit $lret; fi
