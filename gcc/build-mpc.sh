#!/bin/sh
export CFLAGS='-O2 -pedantic -m64 -mtune=k8 -fPIC'
./configure --prefix=$SCPREFIX/gcc \
	--with-gmp=$SCPREFIX/gcc \
	--with-mpfr=$SCPREFIX/gcc \
> configure.log 2>&1
make -j $NP > make.log 2>&1
make check -j $NP > check.log 2>&1
make install > install.log 2>&1
