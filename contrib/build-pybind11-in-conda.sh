#!/bin/bash

# build and install pybind11 for cmake.

conda_root=$(which conda)

if [ -z "$conda_root" ] ; then
  echo "conda not found, exit"
  exit 1
fi

conda_root=$(cd $(dirname "${conda_root}")/.. && pwd)
echo "conda_root: ${conda_root}"

mkdir -p $conda_root/tmp
cd $conda_root/tmp
curl -sSL -o pybind11.zip https://github.com/pybind/pybind11/archive/master.zip
rm -rf pybind11-master
unzip pybind11.zip
cd pybind11-master
mkdir -p build
cd build
cmake -DPYBIND11_TEST=OFF -DPYTHON_EXECUTABLE:FILEPATH=`which python` -DCMAKE_INSTALL_PREFIX=${conda_root} ..
make install
