#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# ${this_script}
WORKSPACE=${WORKSPACE-`pwd`}
retval=0

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/gasdyn

cd $EXHOME/obrf
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/obsq
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/mrefl
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/difr
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/tube
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/blnt
rm -rf result
./go run blnt blnt_m5 blnt_m10 blnt_m20
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
