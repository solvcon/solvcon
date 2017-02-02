#!/bin/bash
#
# Copyright (C) 2017 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download scotch
pkgname=graphviz
pkgver=2.40.1
pkgfull=${pkgname}-${pkgver}
pkgloc=$SCDL/$pkgfull.tar.gz
pkgurl=http://www.graphviz.org/pub/$pkgname/stable/SOURCES/$pkgfull.tar.gz
download $pkgloc $pkgurl 4ea6fd64603536406166600bcc296fc8

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

# build.
if [ $(uname) == Darwin ]; then
{ time ./configure \
  --prefix=$SCDEP \
  --with-quartz \
  ; } > configure.log 2>&1
elif [ $(uname) == Linux ]; then
{ time ./configure \
  --prefix=$SCDEP \
  ; } > configure.log 2>&1
fi
{ time make -j $NP ; } > make.log 2>&1
{ time make install ; } > make.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
