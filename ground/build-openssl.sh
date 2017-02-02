#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDL: downloaded source package file
# - SCDEP: installation destination
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download openssl.
pkgname=openssl
pkgver=1.1.0d
pkgfull=$pkgname-$pkgver
pkgloc=$SCDL/$pkgfull.tar.gz
pkgurl=https://www.openssl.org/source/$pkgfull.tar.gz
download $pkgloc $pkgurl 711ce3cd5f53a99c0e12a7d5804f0f63

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

# build.
{ time ./Configure \
  --prefix=$SCDEP \
  darwin64-x86_64-cc \
  -shared \
; } > configure.log 2>&1
{ time make -j $NP ; } > make.log 2>&1
{ time make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
