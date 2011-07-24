#!/bin/sh

./configure --prefix=$SCPREFIX > configure.log 2>&1
make > make.log 2>&1
