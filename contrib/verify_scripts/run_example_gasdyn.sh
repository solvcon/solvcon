#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# contrib/osumecfd/run_example_gasdyn.sh
WORKSPACE=${WORKSPACE-`pwd`}

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/gasdyn

cd $EXHOME/obrf
rm -rf result
./go run
cd $EXHOME/obsq
rm -rf result
./go run
cd $EXHOME/mrefl
rm -rf result
./go run
cd $EXHOME/difr
rm -rf result
./go run
cd $EXHOME/tube
rm -rf result
./go run
cd $EXHOME/blnt
rm -rf result
./go run blnt blnt_m5 blnt_m10 blnt_m20
