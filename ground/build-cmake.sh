#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download cmake.
pkgname=cmake
pkgverprefix=3.7
pkgver=$pkgverprefix.2
pkgfull=$pkgname-$pkgver
pkgloc=$SCDL/$pkgfull.tar.xz
pkgurl=https://cmake.org/files/v$pkgverprefix/$pkgfull.tar.gz
download $pkgloc $pkgurl 82b143ebbf4514d7e05876bed7a6b1f5

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

# build.
{ time ./configure \
  --prefix=$SCDEP \
; } > configure.log 2>&1
{ time make -j $NP ; } > make.log 2>&1
{ time make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
