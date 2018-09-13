#!/bin/bash
#
# Usage:
#   source <this script>
#   or
#   source <this script> <your-project-folder>
#
SOLVCON_PROJECT=${1:-${HOME}/work-my-projects/solvcon}
SCSRC="${SOLVCON_PROJECT}/solvcon"
SCSRC_WORKING="${SOLVCON_PROJECT}/solvcon-working/"
MINICONDA_DIR="${SCSRC_WORKING}/miniconda"

export PATH="${SCSRC}:${MINICONDA_DIR}/bin/:${PATH}"

mkdir -p ${SCSRC_WORKING}

# fetch SOLVCON source
git clone https://github.com/solvcon/solvcon.git ${SCSRC}

# prepare miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${SCSRC_WORKING}/miniconda.sh

bash ${SCSRC_WORKING}/miniconda.sh -b -p ${MINICONDA_DIR}
conda config --set always_yes yes
conda update -q conda

# create virtual env by conda
conda create -p ${SCSRC_WORKING}/venv-conda --no-default-packages -y python
source activate ${SCSRC_WORKING}/venv-conda

# prepare all packages to build SOLVCON
${SCSRC}/contrib/conda.sh
${SCSRC}/contrib/build-pybind11-in-conda.sh

# begin to build
pushd ${SCSRC}
# build libmarch.so and SOLVCON
make

# test built SOLVCON
echo "======================================"
echo "Start to unit tests and function tests"
echo "======================================"
nosetests --with-doctest
nosetests ftests/gasplus/*

# If unit tests and function tests are good
# install SOLVCON and remove intermediate files
#
# install
make install

echo "Cleaning up intermediate files..."
# clean up intermediate downloaded and built files
make clean
# from build-pybind11-in-conda.sh
ls ${MINICONDA_DIR}
rm -rf ${MINICONDA_DIR}/tmp
rm -rf ${MINICONDA_DIR}/build
rm -rf ${MINICONDA_DIR}/pkgs
# downloaded miniconda
ls ${SCSRC_WORKING}
rm -f ${SCSRC_WORKING}/miniconda.sh
popd

echo "==================="
echo "Completed all tests"
echo "==================="

echo "==================="
echo "SOLVCON PROJECT ENV"
echo "==================="
echo ${SOLVCON_PROJECT}
echo ${PATH}
echo ${PYTHONPATH}
which conda
which python
conda env list

