#!/bin/sh
export PYTHONPATH="../..:$PYTHONPATH"
rm -rf result
mkdir -p mesh
./go run cvg2d_0_0
