#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# contrib/osumecfd/run_example_visout.sh
WORKSPACE=${WORKSPACE-`pwd`}

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/visout

# sequential.
cd $EXHOME/pvtk
rm -rf result
./go run
cd $EXHOME/vtkw
rm -rf result
./go run

# parallel (local).
cd $EXHOME/pvtk
rm -rf result
./go run --npart 3
cd $EXHOME/vtkw
rm -rf result
./go run --npart 3
