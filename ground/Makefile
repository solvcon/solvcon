export SCPREFIX=$(HOME)/opt/scruntime
export SCBIN=$(SCPREFIX)/bin

binary: build/BLAS/make.log build/lapack-3.3.1/make.log build/netcdf-4.1.2/make.log \
	build/libscotchmetis.so \
	build/Python-2.6.6/make.log build/Python-2.7.1/make.log

python: numpy-1.4.1/build

all: binary python

install:
	cd fftw-3.2.1 && make install

build/BLAS/make.log:
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
	cd build/netcdf-4.1.2 && make install > install.log 2>&1

build/libscotchmetis.so:
	mkdir -p build
	cd build && tar xf ../downloaded/scotch_5.1.11.tar.gz
	cd build/scotch_5.1.11/src && \
		cp ../../../scotch_Makefile.inc Makefile.inc && \
		make > make.log 2>&1
	cd build/scotch_5.1.11 && \
		gcc -shared -o ../libscotchmetis.so \
		-Llib -lscotch src/libscotchmetis/*.o

build/Python-2.6.6/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/Python-2.6.6.tar.bz2
	cd build/Python-2.6.6 && ../../build-python.sh
	cd build/Python-2.6.6 && make install > install.log 2>&1
	rm -f $(SCBIN)/python
	mv $(SCBIN)/pydoc $(SCBIN)/pydoc-2.6
	mv $(SCBIN)/idle $(SCBIN)/idle-2.6
	mv $(SCBIN)/2to3 $(SCBIN)/2to3-2.6
	mv $(SCBIN)/smtpd.py $(SCBIN)/smtpd-2.6.py

build/Python-2.7.1/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/Python-2.7.1.tar.bz2
	cd build/Python-2.7.1 && ../../build-python.sh
	cd build/Python-2.7.1 && make install > install.log 2>&1
	rm -f $(SCBIN)/python $(SCBIN)/python-config
	mv $(SCBIN)/pydoc $(SCBIN)/pydoc-2.7
	mv $(SCBIN)/idle $(SCBIN)/idle-2.7
	mv $(SCBIN)/2to3 $(SCBIN)/2to3-2.7
	mv $(SCBIN)/smtpd.py $(SCBIN)/smtpd-2.7.py

numpy-1.4.1/build:
	tar xf downloaded/numpy-1.4.1.tar.gz
	cd numpy-1.4.1 && ../generate-scipy.py
	cd numpy-1.4.1 && $(SCBIN)/python2.6 setup.py build > build2.6.log 2>&1
	cd numpy-1.4.1 && $(SCBIN)/python2.7 setup.py build > build2.7.log 2>&1

clean:
	rm -rf build/BLAS build/lapack-3.3.1 build/netcdf-4.1.2 \
	build/libscotchmetis.so \
	build/Python-2.6.6 build/Python-2.7.1 \
	build/numpy-1.4.1

.PHONY: binary python all clean
