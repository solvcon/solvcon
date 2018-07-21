# This dockerfile is for building solvcon.  To build the image locally, run:
#
# ```
# docker build  .
# ```
#
# You can pull it down using:
#
# ```
# docker pull yungyuc/solvcon_build
# ```


FROM ubuntu:18.04
MAINTAINER Yung-Yu Chen <yyc@solvcon.net>

# Operational necessity.
RUN \
  export DEBIAN_FRONTEND=noninteractive && \
  apt-get -qq update && \
  apt-get -qqy install tzdata && \
  ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
  dpkg-reconfigure --frontend noninteractive tzdata && \
  apt-get -qqy install unzip curl git vim && \
  rm -rf /var/lib/apt/lists/*

# Compiler.
RUN \
  apt-get -qq update && \
  apt-get -qqy install build-essential make cmake libc6-dev gcc-7 g++-7 && \
  rm -rf /var/lib/apt/lists/*

# Math/science tools.
RUN \
  apt-get -qq update && \
  apt-get -qqy install \
    liblapack-dev libhdf5-100 libhdf5-dev libnetcdf13 libnetcdf-dev \
    libscotch-6.0 libscotch-dev gmsh graphviz \
    && \
  rm -rf /var/lib/apt/lists/*

# Python basics.
RUN \
  apt-get -qq update && \
  apt-get -qqy install \
    python3 cython3 python3-numpy python3-nose python3-pytest && \
  rm -rf /var/lib/apt/lists/*

# Python runtime dependencies.
RUN \
  apt-get -qq update && \
  apt-get -qqy install python3-netcdf4 python3-paramiko python3-boto && \
  rm -rf /var/lib/apt/lists/*

# Pybind11.
RUN \
  cd /tmp && \
  curl -sSL -o pybind11.zip https://github.com/pybind/pybind11/archive/master.zip && \
  rm -rf pybind11-master && \
  unzip pybind11.zip && \
  mkdir -p pybind11-master/build && \
  cd pybind11-master/build && \
  cmake -DPYBIND11_TEST=OFF -DPYTHON_EXECUTABLE:FILEPATH=`which python3` -DCMAKE_INSTALL_PREFIX=/usr .. && \
  make install && \
  rm -rf /tmp/pybind11*

# Set up user and environment.
RUN useradd -m solvcon
USER solvcon
ENV HOME="/home/solvcon"
