#!/bin/bash
SCDEV_SRC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SCDEV_ROOT=${SCDEV_SRC}/../../build/env
mkdir -p ${SCDEV_ROOT}
conda create -p ${SCDEV_ROOT}/install --no-default-packages -y python
cp -f ${SCDEV_SRC}/start ${SCDEV_ROOT}/start
