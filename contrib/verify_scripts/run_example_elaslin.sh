#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# contrib/osumecfd/run_example_elaslin.sh
WORKSPACE=${WORKSPACE-`pwd`}

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/elaslin

cd $EXHOME/grpv
rm -rf result
./go run
cd $EXHOME/elcv2d
rm -rf result
./go run elcv2d_200 elcv2d_150 elcv2d_100 elcv2d_50
./go converge --order=2 --stop-on-over
cd $EXHOME/elcv3d
rm -rf result
./go run elcv3d_200 elcv3d_150 elcv3d_100
./go converge --order=2 --stop-on-over
