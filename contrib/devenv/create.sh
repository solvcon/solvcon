#!/bin/bash
SCDEVENV_SRC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -z "$1" ]; then
  SCDEVENV_DST=env
else
  SCDEVENV_DST=$1
fi
SCDEVENV_DST=${SCDEVENV_SRC}/../../build/${SCDEVENV_DST}
echo "Create environment at ${SCDEVENV_DST}"
mkdir -p ${SCDEVENV_DST}
if [ "3" == "`python -c 'import sys; sys.stdout.write(str(sys.version_info.major))'`" ] ; then
  # Fix version to 3.5 before conda Python 3.6 stabilizes
  conda create -p ${SCDEVENV_DST}/install --no-default-packages -y python=3.5
else
  conda create -p ${SCDEVENV_DST}/install --no-default-packages -y python
fi
cp -f ${SCDEVENV_SRC}/start ${SCDEVENV_DST}/start
