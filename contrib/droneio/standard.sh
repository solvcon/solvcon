#!/bin/sh
# install dependency.
contrib/aptget.ubuntu.sh
# build
scons
# unit tests.
nosetests
# functional tests.
nosetests ftests/*
