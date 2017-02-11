#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDL: downloaded source package file
# - SCDEP: installation destination
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download lapack.
lapackname=lapack
lapackver=3.7.0
lapackfull=$lapackname-$lapackver
lapackloc=$SCDL/$lapackfull.tgz
lapackurl=http://www.netlib.org/$lapackname/$lapackname-$lapackver.tgz
download $lapackloc $lapackurl 697bb8d67c7d336a0f339cc9dd0fa72f

# download atlas.
atlasname=atlas
pkgname=atlas
atlasver=3.10.3
atlasfull=$atlasname-$atlasver
atlasloc=$SCDL/$atlasfull.tar.bz2
atlasurl=https://sourceforge.net/projects/math-atlas/files/Stable/$atlasver/$atlasname$atlasver.tar.bz2/download
download $atlasloc $atlasurl d6ce4f16c2ad301837cfb3dade2f7cef

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $atlasloc
rm -rf $atlasfull
mv -f ATLAS $atlasfull
cd $atlasfull

# build.
mkdir -p build
cd build
{ time ../configure \
  --prefix=$SCDEP \
  -Fa alg -fPIC \
  --with-netlib-lapack-tarfile=$lapackloc \
; } > configure.log 2>&1
{ time make ; } > make.log 2>&1
{ time make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
