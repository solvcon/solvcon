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
pkgname=netcdf
pkgver=4.4.1.1
pkgfull=$pkgname-$pkgver
pkgloc=$SCDL/$pkgfull.tar.xz
pkgurl=ftp://ftp.unidata.ucar.edu/pub/$pkgname/$pkgfull.tar.gz
download $pkgloc $pkgurl 503a2d6b6035d116ed53b1d80c811bda

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

# build.
# --with-hdf5 doesn't work:
# http://www.unidata.ucar.edu/support/help/MailArchives/netcdf/msg10457.html
{ time LDFLAGS=-L$SCDEP/lib CPPFLAGS=-I$SCDEP/include ./configure \
  --prefix=$SCDEP \
  --enable-netcdf4 \
  --disable-fortran \
  --disable-dap \
  --enable-shared \
; } > configure.log 2>&1
#  --with-hdf5=$SCDEP \
{ time make -j $NP ; } > make.log 2>&1
{ time make install ; } > install.log 2>&1

# vim: set et nobomb ff=unix fenc=utf8:
