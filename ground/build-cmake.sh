#!/bin/sh

./configure --prefix=$SCROOT > configure.log 2>&1
make -j $NP > make.log 2>&1
make install > install.log 2>&1
