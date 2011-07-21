#!/bin/sh
mkdir -p tmp
tar cfz tmp/ground.tgz ground/*.inc ground/*.sh ground/*.py \
	ground/get* ground/Makefile
