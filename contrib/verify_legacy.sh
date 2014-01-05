#!/bin/sh
# $WORKSPACE is set from jenkins.
# When running from project root manually, use:
# ${this_script}
WORKSPACE=${WORKSPACE-`pwd`}
retval=0

cd $WORKSPACE
for script in `ls $WORKSPACE/contrib/verify_legacy/*.sh`
do
  $script
  lret=$?; if [[ $lret != 0 ]] ; then retval=$lret; fi
done

exit $retval
