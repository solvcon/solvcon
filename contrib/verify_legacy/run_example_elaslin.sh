#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# ${this_script}
WORKSPACE=${WORKSPACE-`pwd`}
retval=0

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/elaslin

cd $EXHOME/grpv
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/elcv2d
rm -rf result
./go run elcv2d_200 elcv2d_150 elcv2d_100 elcv2d_50
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

./go converge --order=2 --stop-on-over
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/elcv3d
rm -rf result
./go run elcv3d_200 elcv3d_150 elcv3d_100
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

./go converge --order=2 --stop-on-over
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
