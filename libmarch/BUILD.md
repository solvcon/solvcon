This is a header-only library and needs no runtime binary.  But the unit tests
require building runtime.  Use [cmake](https://cmake.org) to build the
[googletest](https://github.com/google/googletest) C++ code:

```
$ SRC_PATH=`pwd`
$ BUILD_PATH=/some/place
$ cd ${BUILD_PATH}
$ cmake ${SRC_PATH}
$ cd ${SRC_PATH}
$ cmake -DCMAKE_BUILD_TYPE=Release ${BUILD_PATH}; make -C ${BUILD_PATH} run_gtest
-- Configuring done
-- Generating done
-- Build files have been written to: ${BUILD_PATH}
Scanning dependencies of target test_libmarch
[ 33%] Building CXX object libmarch/tests/CMakeFiles/test_libmarch.dir/main.cpp.o
[ 66%] Building CXX object libmarch/tests/CMakeFiles/test_libmarch.dir/gtest/gtest-all.cc.o
[100%] Linking CXX executable test_libmarch
[100%] Built target test_libmarch
[==========] Running 4 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 4 tests from LookupTableTest
[ RUN      ] LookupTableTest.Sizeof
[       OK ] LookupTableTest.Sizeof (0 ms)
[ RUN      ] LookupTableTest.Construction
[       OK ] LookupTableTest.Construction (0 ms)
[ RUN      ] LookupTableTest.OutOfRange
[       OK ] LookupTableTest.OutOfRange (0 ms)
[ RUN      ] LookupTableTest.WriteCheck
[       OK ] LookupTableTest.WriteCheck (0 ms)
[----------] 4 tests from LookupTableTest (0 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test case ran. (1 ms total)
[  PASSED  ] 4 tests.
[100%] Built target run_gtest
```

If you want to build the debug version, replace ``CMAKE_BUILD_TYPE`` in the the
last command:

```
$ cmake -DCMAKE_BUILD_TYPE=Debug ${BUILD_PATH}; make -C ${BUILD_PATH} 
```
