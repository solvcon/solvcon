#!/bin/bash
SCSRC=${1:-${HOME}/work-my-projects/solvcon/solvcon}
SIMAGE=solvcon.img
# Your home directory in the container.  This would prevent a lot of env issues
# e.g.  conda read a wrong env list.
#
# Besides, this directory is very likely to host your generated data later.
# Create it in a directory which you have permission to write files.
#
# Use
#   singularity --home ${SHOME} ...
# to use this home directory
#
SHOME=`mktemp -d /tmp/singularity-${USER}-XXX`
EXAMPLE_TUBE=${SHOME}/tube
mkdir ${EXAMPLE_TUBE}

# clean before re-building
rm -f ${SIMAGE}

# prepare the SHOME
cp ${SCSRC}/sandbox/gas/tube/go ${EXAMPLE_TUBE}
cp ${SCSRC}/contrib/singularity/sod-tube ${EXAMPLE_TUBE}

# build the image
sudo singularity build $SIMAGE ./Singularity

# run examples
SJOB_COMMAND="singularity exec --home ${SHOME} ${SIMAGE} ${EXAMPLE_TUBE}/sod-tube"

echo ""
echo "======================================================================"
echo "SOLVCON singularity image is built."
echo "Run this command as an example to run a driving script with the image:"
echo ""
echo "${SJOB_COMMAND}"
echo ""
echo "The generated data will be in ${EXAMPLE_TUBE}/result"
echo "======================================================================"

