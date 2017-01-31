#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDL: downloaded source package file
# - SCDEP: installation destination
# - NP: number of processors for compilation
#
# How to check numpy atlas status: http://stackoverflow.com/a/23325759/1805420
# python -c "import numpy.distutils.system_info as si; si.get_info('atlas', 2)"

source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download numpy.
pkgfull=numpy-1.12.0
pkgurl=https://pypi.python.org/packages/b7/9d/8209e555ea5eb8209855b6c9e60ea80119dab5eff5564330b35aa5dc4b2c/$pkgfull.zip
pkgloc=$SCDL/$pkgfull.zip
download $pkgloc $pkgurl 33e5a84579f31829bbbba084fe0a4300

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
unzip -qqo $pkgloc
cd $pkgfull

# build.
rm -f site.cfg
cat > site.cfg << EOF
[DEFAULT]
library_dirs = /usr/local/lib:$SCDEP/lib
include_dirs = /usr/local/include:$SCDEP/include:$SCDEP/include/atlas
[atlas]
atlas_libs = lapack, f77blas, cblas, atlas
EOF

rm -f setup.cfg
cat > setup.cfg << EOF
[config_fc]
fcompiler = gfortran
EOF

{ time $SCDEP/bin/python setup.py build -j $NP ; } > build.log 2>&1

{ time $SCDEP/bin/python setup.py install --old-and-unmanageable ; } > install.log 2>&1

cd $SCDEP
$SCDEP/bin/python -c \
  "import numpy.distutils.system_info as si; si.get_info('atlas', 2)" \
  > $pkgfull/atlas_status.log 2>&1

# vim: set et nobomb ff=unix fenc=utf8:
