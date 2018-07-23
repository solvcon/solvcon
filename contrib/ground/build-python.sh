#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
#
# Note: also download and install pip.
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download python and pip installation script.
mkdir -p $SCDL
curl -sS -o $SCDL/get-pip.py https://bootstrap.pypa.io/get-pip.py
pkgname=Python
pkgver=3.6.0
pkgfull=$pkgname-$pkgver
pkgloc=$SCDL/$pkgfull.tar.xz
pkgurl=https://www.python.org/ftp/python/$pkgver/$pkgfull.tar.xz
download $pkgloc $pkgurl 82b143ebbf4514d7e05876bed7a6b1f5

# unpack.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

# build.
PREFIX=$SCDEP
ARCH=64

mkdir -p $PREFIX/lib
mkdir -p $PREFIX/include

if [ $(uname) == Darwin ]; then
  export CFLAGS="-I$PREFIX/include $CFLAGS"
  export LDFLAGS="-Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -headerpad_max_install_names $LDFLAGS"
  sed -i -e "s/@OSX_ARCH@/$ARCH/g" Lib/distutils/unixccompiler.py
elif [ $(uname) == Linux ]; then
  export CPPFLAGS="-I$PREFIX/include"
  export LDFLAGS="-L$PREFIX/lib -Wl,-rpath=$PREFIX/lib,--no-as-needed"
fi

cfgcmd=("./configure")
cfgcmd+=("--prefix=$PREFIX")
cfgcmd+=("--enable-shared")
cfgcmd+=("--enable-ipv6")
cfgcmd+=("--with-ensurepip=no")
cfgcmd+=("--with-tcltk-includes=-I$PREFIX/include")
cfgcmd+=("--with-tcltk-libs=\"-L$PREFIX/lib -ltcl8.5 -ltk8.5\"")
cfgcmd+=("--enable-loadable-sqlite-extensions")
if [ -n "$SCDEBUG" ] ; then
  cfgcmd+=("--with-pydebug")
fi

echo "${cfgcmd[@]}"
{ time "${cfgcmd[@]}" ; } > configure.log 2>&1
{ time make -j $NP ; } > make.log 2>&1
{ time make install ; } > install.log 2>&1

rm -f $PREFIX/bin/python $PREFIX/bin/pydoc
ln -s $PREFIX/bin/python3.6 $PREFIX/bin/python
ln -s $PREFIX/bin/pydoc3.6 $PREFIX/bin/pydoc

$SCDEP/bin/python $SCDL/get-pip.py

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
