#!/bin/sh
export LD_LIBRARY_PATH=$SCPREFIX/gcc/lib
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$SCPREFIX/gcc/lib
../$1/configure --prefix=$SCPREFIX/gcc \
	--with-gmp=$SCPREFIX/gcc \
	--with-mpfr=$SCPREFIX/gcc \
	--with-mpc=$SCPREFIX/gcc \
	--enable-languages=c,c++,fortran \
	--disable-multilib \
> configure.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1
