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
conda create -p ${SCDEVENV_DST}/install --no-default-packages -y python
cp -f ${SCDEVENV_SRC}/start ${SCDEVENV_DST}/start
