#!/bin/sh

gmsh cube.geo -3 -o ../../tmp/cube.msh -optimize -algo meshadapt
