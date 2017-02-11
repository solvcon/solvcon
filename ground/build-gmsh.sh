#!/bin/bash
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# Consume external variables:
# - SCDEP: installation destination
# - SCDL: downloaded source package file
# - NP: number of processors for compilation
source $(dirname "${BASH_SOURCE[0]}")/scbuildtools.sh

# download gmsh.
pkgname=gmsh
pkgver=2.16.0
pkgfull=${pkgname}-${pkgver}-source
pkgloc=$SCDL/$pkgfull.tgz
pkgurl=http://gmsh.info/src/$pkgfull.tgz
download $pkgloc $pkgurl 762c10f159dab4b042e3140b1c348427

# unpack and patch.
mkdir -p $SCDEP/src
cd $SCDEP/src
tar xf $pkgloc
cd $pkgfull

if [ -n "$(which lsb_release)" ] && [ $(lsb_release -i -s) == Ubuntu ] ; then
cat > patch << EOF
--- CMakeLists.txt  2017-02-11 18:49:01.240546180 +0800
+++ CMakeLists.txt  2017-02-11 20:36:09.556438230 +0800
@@ -346,6 +346,10 @@
         if(LAPACK_LIBRARIES)
           set_config_option(HAVE_BLAS "Blas(ATLAS)")
           set_config_option(HAVE_LAPACK "Lapack(ATLAS)")
+          find_library(GFORTRAN_LIB libgfortran.so.3)
+          if(GFORTRAN_LIB)
+            list(APPEND LAPACK_LIBRARIES \${GFORTRAN_LIB})
+          endif(GFORTRAN_LIB)
         else(LAPACK_LIBRARIES)
           # try with generic names
           set(GENERIC_LIBS_REQUIRED lapack blas pthread)
EOF
patch -p0 < patch
fi

mkdir -p build
cd build

# build.
{ time cmake \
  -DCMAKE_PREFIX_PATH=$SCDEP \
  -DCMAKE_INSTALL_PREFIX=$SCDEP \
  -DENABLE_NUMPY=ON \
  -DENABLE_OS_SPECIFIC_INSTALL=OFF \
  -DENABLE_MATCH=OFF \
  -DENABLE_PETSC=OFF \
  -DENABLE_SLEPC=OFF \
  .. ; } > cmake.log 2>&1
{ make -j $NP VERBOSE=1 ; } > make.log 2>&1
{ make install ; } > install.log 2>&1

# finalize.
finalize $pkgname

# vim: set et nobomb ff=unix fenc=utf8:
