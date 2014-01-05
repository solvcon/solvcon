#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# ${this_script}
WORKSPACE=${WORKSPACE-`pwd`}
retval=0

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/vslin

cd $EXHOME/grpv
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/cvg
rm -rf result
./go run cvg2d_200 cvg2d_150 cvg2d_100 cvg2d_50 cvg3d_200 cvg3d_150 cvg3d_100
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

./go converge --order=2 --stop-on-over
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
