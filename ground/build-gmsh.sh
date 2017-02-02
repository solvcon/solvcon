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
pkgname=gmsh
pkgver=2.16.0
pkgfull=${pkgname}-${pkgver}-source
pkgloc=$SCDL/$pkgfull.tgz
pkgurl=http://gmsh.info/src/$pkgfull.tgz
download $pkgloc $pkgurl 762c10f159dab4b042e3140b1c348427

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull
mkdir -p build
cd build

# build.
{ time cmake \
  -DCMAKE_PREFIX_PATH=$SCDEP \
  -DCMAKE_INSTALL_PREFIX=$SCDEP \
  -DENABLE_NUMPY=ON \
  -DENABLE_OS_SPECIFIC_INSTALL=OFF \
  -DENABLE_MATCH=OFF \
  -DENABLE_PETSC=OFF \
  -DENABLE_SLEPC=OFF \
  .. ; } > cmake.log 2>&1
{ make -j $NP ; } > make.log 2>&1
{ make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
