#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download netcdf.
pkgname=hdf5
pkgver=1.8.18
pkgfull=$pkgname-$pkgver
pkgloc=$SCDL/$pkgfull.tar.bz2
pkgurl=https://support.hdfgroup.org/ftp/HDF5/current18/src/$pkgfull.tar.bz2
download $pkgloc $pkgurl 29117bf488887f89888f9304c8ebea0b

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
{ time make install ; } > install.log 2>&1

# vim: set et nobomb ff=unix fenc=utf8:
