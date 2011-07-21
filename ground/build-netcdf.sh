#!/bin/sh

./configure --prefix=$SCPREFIX \
	--disable-fortran \
	--disable-dap \
	--enable-shared \
> configure.log 2>&1
	#--with-hdf5=$HDF5_HOME \
	#--enable-netcdf4 \

make > make.log 2>&1
