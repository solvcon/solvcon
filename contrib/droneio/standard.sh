#!/bin/sh
# install dependency.
contrib/aptget.ubuntu.sh
# build
scons
# unit tests.
nosetests -v
# functional tests.
nosetests ftests/* --exclude test_rpc --exclude test_remote -v
