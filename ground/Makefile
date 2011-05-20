all: build/BLAS/make.log build/lapack-3.3.1/make.log build/netcdf-4.1.2/make.log \
	build/libscotchmetis.so \
	Python-2.6.6/make.log Python-2.7.1/make.log

install:
	cd fftw-3.2.1 && make install

biuld/BLAS/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/blas.tgz
	cd build/BLAS && ../../build-blas.sh

build/lapack-3.3.1/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/lapack-3.3.1.tgz
	cd build/lapack-3.3.1 && ../../build-lapack.sh

build/netcdf-4.1.2/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/netcdf-4.1.2.tar.gz
	cd build/netcdf-4.1.2 && ../../build-netcdf.sh

build/libscotchmetis.so:
	mkdir -p build
	cd build && tar xf ../downloaded/scotch_5.1.11.tar.gz
	cd build/scotch_5.1.11/src && \
		ln -sf ../../../scotch_Makefile.inc Makefile.inc && \
		make > make.log 2>&1
	cd build/scotch_5.1.11 && \
		gcc -shared -o ../libscotchmetis.so \
		-Llib -lscotch src/libscotchmetis/*.o

Python-2.6.6/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/Python-2.6.6.tar.bz2
	cd build/Python-2.6.6 && ../../build-python.sh

Python-2.7.1/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/Python-2.7.1.tar.bz2
	cd build/Python-2.7.1 && ../../build-python.sh

clean:
	rm -rf build/BLAS build/lapack-3.3.1 build/netcdf-4.1.2 \
	build/libscotchmetis.so \
	Python-2.6.6 Python-2.7.1

.PHONY: clean
