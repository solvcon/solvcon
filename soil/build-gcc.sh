#!/bin/sh
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# Building issues:
# * asm/errno.h on Ubuntu 11.04:
#   https://bugs.launchpad.net/ubuntu/+source/clang/+bug/774215

PKGNAME=$1
if [ -z "$PKGNAME" ]
  then
    echo "PKGNAME (parameter 1) not set"
    exit
fi

# unpack.
mkdir -p $TMPBLD
cd $TMPBLD
tar xfj ../$TMPDL/$PKGNAME.tar.bz2

# build.
mkdir -p $PKGNAME-build
cd $PKGNAME-build
export LD_LIBRARY_PATH=$SCROOT/soil/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SCROOT/soil/lib
#export CFLAGS=-m64
../$PKGNAME/configure --prefix=$SCROOT/soil \
    $GCCOPTS \
	--with-gmp=$SCROOT/soil \
	--with-mpfr=$SCROOT/soil \
	--with-mpc=$SCROOT/soil \
	--enable-languages=c,c++,fortran \
	--disable-multilib \
> configure.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1

# vim: set ai et nu:
