# This dockerfile builds a docker image for building doriath with gitlab ci:
# https://gitlab.com/solvcon/doriath/pipelines .
#
# To build the image locally, run:
#
# ```
# docker build  .
# ```
#
# The public repository for this image is at:
# https://hub.docker.com/r/yungyuc/doriath_build/ .  You can pull it down
# using:
#
# ```
# docker pull yungyuc/doriath_build
# ```


FROM ubuntu:18.04
MAINTAINER Yung-Yu Chen <yyc@solvcon.net>

ARG doriath_repo=https://gitlab.com/solvcon/doriath
ARG doriath_branch=master

# Install OS-wide packages.
RUN \
  apt-get -qq update && \
  apt-get -qqy install build-essential g++ liblapack-dev curl git unzip && \
  rm -rf /var/lib/apt/lists/*

# Set up user and environment.
RUN useradd -m doriath
USER doriath
ENV HOME="/home/doriath"
ENV CONDA_PREFIX="$HOME/opt/conda3"
ENV PATH="${CONDA_PREFIX}/bin:$PATH"

# Install and update miniconda.
RUN \
  curl -sSL -o /tmp/miniconda3.sh \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  bash /tmp/miniconda3.sh -b -p ${CONDA_PREFIX} && \
  conda config --set always_yes yes --set changeps1 no && \
  conda update --quiet conda && \
  conda update --all --quiet && \
  conda clean --all --quiet --yes && \
  rm -rf /tmp/miniconda3.sh ${CONDA_PREFIX}/pkgs/* ${CONDA_PREFIX}/tmp/*

# Install build-necessary dependencies.
RUN \
  cd /tmp && \
  doriath_src="/tmp/doriath" && \
  git clone -b ${doriath_branch} ${doriath_repo} ${doriath_src} && \
  ${doriath_src}/contrib/conda.sh && \
  ${doriath_src}/contrib/build-pybind11-in-conda.sh && \
  conda clean --all --quiet --yes && \
  rm -rf ${doriath_src} ${CONDA_PREFIX}/pkgs/* ${CONDA_PREFIX}/tmp/*
