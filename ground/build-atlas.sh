#!/bin/bash

LAPACK=`pwd`/../lapack-3.3.1/lapack_LINUX.a
ATLAS_PLAT=ATLAS_LINUX

# determine arch.
if [ `uname -m` == 'x86_64' ]; then
  BITS='-b 64'
else
  BITS='-b 32'
fi

mkdir -p $ATLAS_PLAT
cd $ATLAS_PLAT
../configure --prefix=$SCROOT \
	-Si cputhrchk 0 \
	$BITS \
	-Fa alg -fPIC \
	--with-netlib-lapack=$LAPACK \
> configure.log 2>&1

make > make.log 2>&1
make install > install.log 2>&1
