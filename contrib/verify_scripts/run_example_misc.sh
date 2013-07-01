#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# contrib/osumecfd/run_example_misc.sh
WORKSPACE=${WORKSPACE-`pwd`}

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/misc

cd $EXHOME/elas3d
scons
rm -rf result
./go run
