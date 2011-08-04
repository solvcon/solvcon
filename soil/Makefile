export SCROOT?=$(HOME)/opt/scruntime
export SCBIN=$(SCROOT)/bin
export NP?=$(`cat /proc/cpuinfo | grep processor | wc -l`)

default: gcc

gcc: build/gmp-5.0.2/make.log build/mpfr-3.0.1/make.log build/mpc-0.9/make.log \
	build/gcc-4.6.1-build/make.log \
	$(SCBIN)/scgccvars.sh

build/gmp-5.0.2/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/gmp-5.0.2.tar.bz2
	cd build/gmp-5.0.2 && ../../build-gmp.sh

build/mpfr-3.0.1/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/mpfr-3.0.1.tar.bz2
	cd build/mpfr-3.0.1 && patch -N -Z -p1 < ../../downloaded/allpatches > \
		patch.log 2>&1
	cd build/mpfr-3.0.1 && ../../build-mpfr.sh

build/mpc-0.9/make.log:
	mkdir -p build
	cd build && tar xf ../downloaded/mpc-0.9.tar.gz
	cd build/mpc-0.9 && ../../build-mpc.sh

build/gcc-4.6.1-build/make.log:
	mkdir -p build build/gcc-4.6.1-build
	cd build && tar xf ../downloaded/gcc-4.6.1.tar.bz2
	cd build/gcc-4.6.1-build && ../../build-gcc.sh gcc-4.6.1

$(SCBIN)/scgccvars.sh:
	./create-scgccvars.sh

clean:
	rm -rf build/gmp-5.0.2 build/mpfr-3.0.1 build/mpc-0.9 \
	build/gcc-4.6.1-build

.PHONY: binary python all clean
