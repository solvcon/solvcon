export SCPREFIX?=$(HOME)/opt/scruntime
export SCBIN=$(SCPREFIX)/bin
export NP?=$(`cat /proc/cpuinfo | grep processor | wc -l`)

binary: build/BLAS/make.log build/lapack-3.3.1/make.log \
	build/ATLAS/ATLAS_LINUX/make.log \
	build/netcdf-4.1.2/make.log build/libscotchmetis.so \
	build/Python-2.6.6/make.log build/Python-2.7.1/make.log \
	$(SCBIN)/scvars.sh

python: build/mercurial-1.9/build \
	build/numpy-1.6.0/build build/scons-2.0.1/build \
	build/nose-1.0.0/build build/epydoc-3.0.1/build

vtk: build/cmake-2.8.5/make.log build/vtk-5.6.1/make.log

all: binary python vtk

build/BLAS/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/blas.tgz
	cd build/BLAS && ../../build-blas.sh

build/lapack-3.3.1/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/lapack-3.3.1.tgz
	cd build/lapack-3.3.1 && ../../build-lapack.sh

build/ATLAS/ATLAS_LINUX/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/atlas3.8.4.tar.bz2
	cd build/ATLAS && ../../build-atlas.sh

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

$(SCBIN)/scvars.sh:
	$(SCBIN)/python2.7 create-scvars.py

build/mercurial-1.9/build:
	mkdir -p build
	cd build && tar xf ../downloaded/mercurial-1.9.tar.gz
	cd build/mercurial-1.9 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1

build/numpy-1.6.0/build:
	mkdir -p build
	cd build && tar xf ../downloaded/numpy-1.6.0.tar.gz
	cd build/numpy-1.6.0 && ../../generate-numpy.py
	cd build/numpy-1.6.0 && \
		$(SCBIN)/python2.6 setup.py install > install2.6.log 2>&1
	cd build/numpy-1.6.0 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1

build/scons-2.0.1/build:
	mkdir -p build
	cd build && tar xf ../downloaded/scons-2.0.1.tar.gz
	cd build/scons-2.0.1 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/scons*

build/nose-1.0.0/build:
	mkdir -p build
	cd build && tar xf ../downloaded/nose-1.0.0.tar.gz
	cd build/nose-1.0.0 && \
		$(SCBIN)/python2.6 setup.py install > install2.6.log 2>&1
	cd build/nose-1.0.0 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/nosetests*

build/epydoc-3.0.1/build:
	mkdir -p build
	cd build && tar xf ../downloaded/epydoc-3.0.1.tar.gz
	cd build/epydoc-3.0.1 && \
		$(SCBIN)/python2.6 setup.py install > install2.6.log 2>&1
	cd build/epydoc-3.0.1 && \
		$(SCBIN)/python2.7 setup.py install > install2.7.log 2>&1
	$(SCBIN)/python2.7 modify-pyshebang.py $(SCBIN)/epydoc*

build/cmake-2.8.5/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/cmake-2.8.5.tar.gz
	cd build/cmake-2.8.5 && ../../build-cmake.sh

build/vtk-5.6.1/make.log:
	mkdir -p build/vtk-5.6.1
	cd build && tar xf ../downloaded/vtk-5.6.1.tar.gz
	cd build && tar xf ../downloaded/vtkdata-5.6.1.tar.gz
	cd build/vtk-5.6.1 && ../../build-vtk.sh

clean:
	rm -rf build/BLAS build/lapack-3.3.1 build/netcdf-4.1.2 \
	build/ATLAS build/scotch_5.8.11 build/libscotchmetis.so \
	build/Python-2.6.6 build/Python-2.7.1 $(SCBIN)/scvars.sh \
	build/mercurial-1.9/ \
	build/numpy-1.6.0 build/scons-2.0.1 build/nose-1.0.0 \
	build/epydoc-3.0.1 \
	build/cmake-2.8.5 build/vtk-5.6.1

.PHONY: binary python all clean
