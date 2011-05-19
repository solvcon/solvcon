#!/bin/sh

./configure --prefix=$HOME/opt \
	--disable-fortran \
	--enable-shared \
> configure.log 2>&1
	#--with-hdf5=$HDF5_HOME \
	#--enable-netcdf4 \

make > make.log 2>&1
