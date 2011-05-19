all: BLAS/make.log lapack-3.3.1/make.log netcdf-4.1.2/make.log

install:
	cd fftw-3.2.1 && make install

BLAS/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/blas.tgz
	cd build/BLAS && ../../build-blas.sh

lapack-3.3.1/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/lapack-3.3.1.tgz
	cd build/lapack-3.3.1 && ../../build-lapack.sh

netcdf-4.1.2/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/netcdf-4.1.2.tar.gz
	cd build/netcdf-4.1.2 && ../../build-netcdf.sh

clean:
	rm -rf build/BLAS build/lapack-3.3.1 build/netcdf-4.1.2

.PHONY: clean
