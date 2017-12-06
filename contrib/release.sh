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
nosetests --with-doctest ; lret=$?
if [[ $lret != 0 ]] ; then retval=$lret ; fi
nosetests ftests/gasplus/* -v ; lret=$?
if [[ $lret != 0 ]] ; then retval=$lret ; fi
nosetests ftests/parallel/* ; lret=$?
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  echo "As of 2017/12/6, travis osx image randomly fails tests related to rpc \
with ssh.  Disregard the error report for now."
else
  if [[ $lret != 0 ]] ; then retval=$lret ; fi
fi

exit $retval
