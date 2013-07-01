#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# contrib/osumecfd/run_example_euler.sh
WORKSPACE=${WORKSPACE-`pwd`}

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/euler

cd $EXHOME/hbnt
rm -rf result
./go run
cd $EXHOME/obrefl
rm -rf result
./go run
cd $EXHOME/impl
rm -rf result
./go run
