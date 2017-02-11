#!/bin/bash

anaconda=Miniconda3-latest-Linux-x86_64.sh
workdir=/vagrant/tmp

mkdir -p $workdir
cd $workdir

if [[ ! -f $workdir/$anaconda ]]; then
  wget --quiet http://repo.continuum.io/miniconda/$anaconda -O $workdir/$anaconda
fi

if [[ `which conda` != "${HOME}/opt/conda3/bin/conda" ]]; then
  bash $workdir/$anaconda -b -p ${HOME}/opt/conda3
fi

# vim: set et nobomb fenc=utf8 ft=sh ff=unix sw=2 ts=2:
