#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# contrib/osumecfd/run_example_vslin.sh
WORKSPACE=${WORKSPACE-`pwd`}

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/vslin

cd $EXHOME/grpv
rm -rf result
./go run
cd $EXHOME/cvg
rm -rf result
./go run cvg2d_200 cvg2d_150 cvg2d_100 cvg2d_50 cvg3d_200 cvg3d_150 cvg3d_100
./go converge --order=2 --stop-on-over
