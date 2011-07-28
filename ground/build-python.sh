#!/bin/sh

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
