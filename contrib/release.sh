#!/bin/sh
SCVER=`python -c 'import solvcon; print solvcon.__version__'`
# build source distribution file.
rm -rf build/ lib/
python2.7 setup.py sdist
# build scdata.
tar cfz dist/scdata-$SCVER.tgz scdata/mesh/*.neu.gz
# build the project.
cd dist
tar xfz SOLVCON-$SCVER.tar.gz
cd SOLVCON-$SCVER
tar xfz ../scdata-$SCVER.tgz
scons --download --extract
# test.
nosetests
