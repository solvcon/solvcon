#!/usr/bash
cc -I $PREFIX/include -L $PREFIX/lib test.c -lgmp -o test.out

if [ `uname` == Darwin ]; then
  env DYLD_LIBRARY_PATH=$PREFIX/lib ./test.out
else
  env LD_LIBRARY_PATH=$PREFIX/lib ./test.out
fi
