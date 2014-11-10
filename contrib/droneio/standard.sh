#!/bin/sh
# install dependency.
sudo apt-get update
contrib/aptget.ubuntu.12.04LTS.sh

# initialize return value.
retval=0
# build with scons.
scons
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi
# unit tests.
nosetests -v
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi
# functional tests.
nosetests ftests/* --exclude test_rpc --exclude test_remote -v
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi
# remove generated binaries.
find solvcon -name *.so -delete
rm -rf build/ lib/
# rebuild with distutils.
python setup.py build_ext --inplace
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi
# unit tests.
nosetests -v
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi
# functional tests.
nosetests ftests/* --exclude test_rpc --exclude test_remote -v
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
