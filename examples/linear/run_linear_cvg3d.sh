#!/bin/bash
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:

# environment settings.
WORKSPACE=${WORKSPACE-`pwd`}
PYBIN=`which python2.7`
## PYTHONPATH
if [[ $1 == "local" ]]; then
  export PYTHONPATH=$WORKSPACE:$OLD_PYTHONPATH
else
  OLD_PYTHONPATH=$PYTHONPATH
  export PYTHONPATH=$WORKSPACE:$OLD_PYTHONPATH
  scvar=`$PYBIN -c 'import solvcon; print solvcon.__version__'`
  export PYTHONPATH=$WORKSPACE/dist/SOLVCON-$scvar:$OLD_PYTHONPATH
  unset OLD_PYTHONPATH scvar
fi
## example home.
EXHOME=$WORKSPACE/examples/linear/cvg
RESULTDIR=$EXHOME/result
## initialize return value (no error).
retval=0

# change to location and clear left-overs.
cd $EXHOME
rm -rf $RESULTDIR

# 3D.
cmd="$PYBIN go run cvg3d_400 cvg3d_200 cvg3d_150 cvg3d_100"
echo $cmd
$cmd
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

# print converge.
cmd="$PYBIN go converge cvg3d --order=2 --order-tolerance=0.5 --stop-on-over"
echo $cmd
$cmd
lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi

exit $retval
