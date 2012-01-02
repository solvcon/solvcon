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

PKGNAME=$1
if [ -z "$PKGNAME" ]
  then
    echo "PKGNAME (parameter 1) not set"
    exit
fi

# unpack.
mkdir -p $TMPBLD
cd $TMPBLD
tar xfz ../$TMPDL/$PKGNAME.tar.gz
tar xfz ../$TMPDL/$2.tar.gz

# build.
mkdir -p $PKGNAME
cd $PKGNAME
cmake \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -fPIC" \
    -DCMAKE_C_FLAGS="-O3 -DNDEBUG -fPIC" \
    -DCMAKE_INSTALL_PREFIX=$SCROOT \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DVTK_WRAP_PYTHON=ON \
    -DVTK_USE_TK=OFF \
    -DPYTHON_EXECUTABLE=$SCBIN/python2.7 \
    -DPYTHON_INCLUDE_DIR=$SCROOT/include/python2.7 \
    -DPYTHON_LIBRARY=$SCROOT/lib/libpython2.7.so \
    ../VTK > cmake.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1

# vim: set ai et nu:
