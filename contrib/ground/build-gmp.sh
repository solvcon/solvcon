#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download gmsh.
pkgname=gmp
pkgver=6.1.2
pkgfull=${pkgname}-${pkgver}
pkgloc=$SCDL/$pkgfull.tar.bz2
pkgurl=https://gmplib.org/download/$pkgname/$pkgfull.tar.bz2
download $pkgloc $pkgurl 8ddbb26dc3bd4e2302984debba1406a5

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

# build.
{ time ./configure \
  --prefix=$SCDEP \
  --enable-cxx \
  ; } > configure.log 2>&1
{ time make -j $NP ; } > make.log 2>&1
{ time make check ; } > check.log 2>&1
{ time make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
