#!/bin/sh
SCVER=`python -c 'import sys; import solvcon; sys.stdout.write("%s\\n" % solvcon.__version__)'`
# build source distribution file.
rm -rf dist/SOLVCON-${SCVAR}*
python setup.py clean
python setup.py sdist
# build the project.
cd dist
rm -rf SOLVCON-${SCVER}/
tar xfz SOLVCON-${SCVER}.tar.gz
cd SOLVCON-${SCVER}
python setup.py build_ext --inplace
# test.
nosetests --with-doctest
