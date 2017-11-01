#!/bin/bash -x

SCVER=`python -c 'import sys; import solvcon; sys.stdout.write("%s\\n" % solvcon.__version__)'`

# package source distribution file.
rm -rf dist/SOLVCON-${SCVER}*
python setup.py clean
python setup.py sdist

# unpack the distribution file.
cd dist
rm -rf SOLVCON-${SCVER}/
tar xfz SOLVCON-${SCVER}.tar.gz
cd SOLVCON-${SCVER}

retval=0

# build.
python setup.py build_ext --inplace
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret ; fi

# test.
PYTHONPATH=`pwd`
nosetests --with-doctest
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret ; fi
nosetests ftests/gasplus/* -v
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret ; fi
nosetests ftests/parallel/*
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret ; fi

exit $retval
