#!/bin/sh
PYTHONPATH=../../..:$PYTHONPATH
rm -rf result
# 2D.
./go run cvg2d_200 cvg2d_150 cvg2d_100 cvg2d_50 
# 3D.
./go run cvg3d_500 cvg3d_400 cvg3d_200 cvg3d_150 cvg3d_100
# print converge.
./go converge
