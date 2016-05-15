#!/bin/bash
GOOGLE_TEST_SRC="$1" # git clone http://github.com/google/googletest
FUSE_BIN="${GOOGLE_TEST_SRC}/googletest/scripts/fuse_gtest_files.py"
${FUSE_BIN} .
