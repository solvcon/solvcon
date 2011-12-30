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
tar xfj ../$TMPDL/$PKGNAME.tar.bz2

# build.
cd $PKGNAME
./configure --prefix=$SCROOT \
	--enable-shared \
	--enable-ipv6 \
	--enable-unicode=ucs4 \
	--without-cxx \
	--with-system-ffi \
	--with-fpectl \
	LDFLAGS=-Wl,-rpath=$SCROOT/lib \
> configure.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1
rm -f $SCBIN/python $SCBIN/python-config
mv $SCBIN/pydoc $SCBIN/pydoc-2.7
mv $SCBIN/idle $SCBIN/idle-2.7
mv $SCBIN/2to3 $SCBIN/2to3-2.7
mv $SCBIN/smtpd.py $SCBIN/smtpd-2.7.py

# vim: set ai et nu:
