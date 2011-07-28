#!/bin/sh
cp ../../blas.make.inc make.inc
make -j $NP > make.log 2>&1
