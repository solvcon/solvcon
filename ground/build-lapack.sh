#!/bin/sh

sed -e "s/OPTS     \= -O2/OPTS     \=\ -O2\ -fPIC/g" \
	INSTALL/make.inc.gfortran | \
	sed -e "s/NOOPT    \= -O0/NOOPT    \= -O0 -fPIC/g" > \
	make.inc

cd SRC
make > ../make.log 2>&1
