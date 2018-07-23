#!/bin/bash -x

SCVER=$(python3 -c 'import sys; import solvcon; sys.stdout.write("%s"%solvcon.__version__)')

# package source distribution file.
rm -rf dist/SOLVCON-${SCVER}*
python3 setup.py clean
python3 setup.py sdist

# unpack the distribution file.
if [[ ! -d "dist" ]] ; then
  echo "fatal error: dist doesn't exist"
  exit 1
fi
cd dist
rm -rf SOLVCON-${SCVER}/
tar xfz SOLVCON-${SCVER}.tar.gz
cd SOLVCON-${SCVER}

retval=0

# build.
make
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret ; fi

# unit test.
if [[ -n `which nosetests3` ]] ; then
  NOSETESTS=nosetests3
else
  NOSETESTS=nosetests
fi
PYTHONPATH=`pwd`
$NOSETESTS --with-doctest ; lret=$?
if [[ $lret != 0 ]] ; then retval=$lret ; fi

exit $retval
