#!/bin/sh
# Building issues:
# * asm/errno.h on Ubuntu 11.04:
#   https://bugs.launchpad.net/ubuntu/+source/clang/+bug/774215

export LD_LIBRARY_PATH=$SCROOT/gcc/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SCROOT/gcc/lib
#export CFLAGS=-m64
../$1/configure --prefix=$SCROOT/gcc \
	--with-gmp=$SCROOT/gcc \
	--with-mpfr=$SCROOT/gcc \
	--with-mpc=$SCROOT/gcc \
	--enable-languages=c,c++,fortran \
	--disable-multilib \
> configure.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1
