#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download scotch
pkgname=scotch
pkgver=6.0.4
pkgfull=${pkgname}_${pkgver}
pkgloc=$SCDL/$pkgfull.tar.gz
pkgurl=http://gforge.inria.fr/frs/download.php/file/34618/$pkgfull.tar.gz
download $pkgloc $pkgurl d58b825eb95e1db77efe8c6ff42d329f

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull
cd src # this is the working directory.

# patch.
if [ $(uname) == Darwin ]; then
  patch -p1 < $SCGROUND/scotch_clock_gettime.patch
fi

# build.
echo "prefix = $SCDEP" > Makefile.inc
echo '' >> Makefile.inc
if [ $(uname) == Darwin ]; then
  cat $SCGROUND/scotch_Makefile_darwin.inc >> Makefile.inc
elif [ $(uname) == Linux ]; then
  cat Make.inc/Makefile.inc.x86-64_pc_linux2 | \
    sed -e "s/= -O3/= -fPIC -O3/" >> \
    Makefile.inc
fi
{ time make -j $NP ; } > make.log 2>&1
cd ..

# install.
mkdir -p $SCDEP/lib
cp lib/* $SCDEP/lib
mkdir -p $SCDEP/bin
cp bin/* $SCDEP/bin
mkdir -p $SCDEP/include
cp include/* $SCDEP/include

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
