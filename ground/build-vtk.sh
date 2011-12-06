#!/bin/sh

cmake \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -fPIC" \
    -DCMAKE_C_FLAGS="-O3 -NDEBUG -fPIC" \
    -DCMAKE_INSTALL_PREFIX=$SCROOT \
    -DBUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DVTK_WRAP_PYTHON=ON \
    -DVTK_USE_TK=OFF \
    -DPYTHON_EXECUTABLE=$SCBIN/python2.7 \
    -DPYTHON_INCLUDE_DIR=$SCROOT/include/python2.7 \
    -DPYTHON_LIBRARY=$SCROOT/lib/libpython2.7.so \
    ../VTK > cmake.log 2>&1

make -j $NP > make.log 2>&1
make install > install.log 2>&1
