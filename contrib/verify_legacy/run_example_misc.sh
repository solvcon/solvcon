#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# ${this_script}
WORKSPACE=${WORKSPACE-`pwd`}
retval=0

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/misc

cd $EXHOME/elas3d
scons
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
