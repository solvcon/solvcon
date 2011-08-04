#!/bin/sh
# Building issues:
# * asm/errno.h on Ubuntu 11.04:
#   https://bugs.launchpad.net/ubuntu/+source/clang/+bug/774215

export LD_LIBRARY_PATH=$SCROOT/soil/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SCROOT/soil/lib
#export CFLAGS=-m64
../$1/configure --prefix=$SCROOT/soil \
	--with-gmp=$SCROOT/soil \
	--with-mpfr=$SCROOT/soil \
	--with-mpc=$SCROOT/soil \
	--enable-languages=c,c++,fortran \
	--disable-multilib \
> configure.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1
