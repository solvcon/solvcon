# This dockerfile builds a docker image containing the latest SOLVCON on git.
# It can be used to quickly set up SOLVCON in docker.
# 
# Go to where this dockerfile is and run:
# 
# ```
# docker build -t <docker_repository>:<docker_tag> .
# ```
# 
# And then verify it with
# 
# ```
# docker run <docker_repository>:<docker_tag> bash -c "source /home/solvcon/opt/conda3/bin/activate; nosetests --with-doctest /home/solvcon/solvcon/"
# ```
# 
# If any unit test fails, there's a problem.

FROM ubuntu:14.04
MAINTAINER Taihsiang Ho <tai271828@gmail.com>
# Install OS-wide packages.
RUN \
  apt-get -qq update && \
  apt-get -qqy install g++ liblapack-dev git wget
# Set up user and environment.
RUN useradd -m solvcon
USER solvcon
ENV HOME="/home/solvcon"
ENV CONDA_PREFIX="$HOME/opt/conda3"
ENV PATH="${CONDA_PREFIX}/bin:$PATH"
ENV SCSRC="$HOME/solvcon"
# Get latest SOLVCON from github.
RUN cd $HOME && git clone https://github.com/solvcon/solvcon.git
# Install most dependencies at user level through conda and pip.
RUN \
  mkdir -p $HOME/tmp && \
  cd $HOME && \
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
  bash /tmp/miniconda3.sh -b -p ${CONDA_PREFIX} && \
  rm -f /tmp/miniconda3.sh
RUN \
  conda config --set always_yes yes --set changeps1 no && \
  conda update -q conda && \
  $SCSRC/contrib/conda.sh && \
  pip install https://github.com/pybind/pybind11/archive/master.zip
# Build SOLVCON.
RUN cd $SCSRC && python setup.py build_ext --inplace
