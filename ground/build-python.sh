#!/bin/sh

ldflag='-Wl,-rpath '$HOME/opt/lib

./configure --prefix=$HOME/opt \
	--enable-shared \
	--enable-ipv6 \
	--enable-unicode=ucs4 \
	--without-cxx \
	--with-system-ffi \
	--with-fpectl \
	LDFLAGS=-Wl,-rpath=$HOME/opt/lib \
> configure.log 2>&1

make > make.log 2>&1
