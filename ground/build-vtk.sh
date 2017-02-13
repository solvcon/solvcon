#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download vtk.
pkgname=VTK
pkgverprefix=7.1
pkgver=$pkgverprefix.0
pkgfull=$pkgname-$pkgver
pkgloc=$SCDL/$pkgfull.tar.xz
pkgurl=http://www.vtk.org/files/release/$pkgverprefix/$pkgfull.tar.gz
download $pkgloc $pkgurl a7e814c1db503d896af72458c2d0228f

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
mkdir -p $pkgfull-build
cd $pkgfull-build

# build.
if [ -n "$SCDEBUG" ] ; then
  pybin=python3.6dm
  buildtype=Debug
else
  pybin=python3.6m
  buildtype=Release
fi
{ time cmake \
  -DCMAKE_BUILD_TYPE=$buildtype \
  -DCMAKE_PREFIX_PATH=$SCDEP \
  -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -fPIC" \
  -DCMAKE_C_FLAGS="-O3 -DNDEBUG -fPIC" \
  -DCMAKE_INSTALL_PREFIX=$SCDEP \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DVTK_WRAP_PYTHON=ON \
  -DVTK_USE_TK=OFF \
  -DPYTHON_EXECUTABLE=$SCDEP/bin/python3.6 \
  -DPYTHON_INCLUDE_DIR=$SCDEP/include/$pybin \
  -DPYTHON_LIBRARY=$SCDEP/lib/lib$pybin.$SCDLLEXT \
  ../$pkgfull \
; } > cmake.log 2>&1
{ time make -j $NP ; } > make.log 2>&1
{ time make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
