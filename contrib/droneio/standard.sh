#!/bin/sh
# install dependency.
sudo apt-get update
contrib/aptget.ubuntu.12.04LTS.sh
# build with scons.
scons
# unit tests.
nosetests -v
# functional tests.
nosetests ftests/* --exclude test_rpc --exclude test_remote -v
# remove generated binaries.
find solvcon -name *.so -delete
rm -rf build/ lib/
# rebuild with distutils.
python setup.py build_ext --inplace
# unit tests.
nosetests -v
# functional tests.
nosetests ftests/* --exclude test_rpc --exclude test_remote -v
