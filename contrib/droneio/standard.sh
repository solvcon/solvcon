#!/bin/sh
# install dependency.
contrib/aptget.ubuntu.12.04LTS.sh
# build
scons
# unit tests.
nosetests -v
# functional tests.
nosetests ftests/* --exclude test_rpc --exclude test_remote -v
