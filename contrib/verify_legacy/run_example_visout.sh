#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# scons
# scons --get-scdata # require internet connection.
# ${this_script}
WORKSPACE=${WORKSPACE-`pwd`}
retval=0

export PYTHONPATH=$WORKSPACE:$PYTHONPATH
export EXHOME=$WORKSPACE/examples/visout

# sequential.
cd $EXHOME/pvtk
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/vtkw
rm -rf result
./go run
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

# parallel (local).
cd $EXHOME/pvtk
rm -rf result
./go run --npart 3
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

cd $EXHOME/vtkw
rm -rf result
./go run --npart 3
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
