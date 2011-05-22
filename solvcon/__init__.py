# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
SOLVCON: a multi-physics, supercomputing software framework for high-fidelity
solutions of partial differential equations (PDEs) by hybrid parallelism.

SOLVCON facilitates rapid devlopment of PDE solvers for massively parallel
computing.  C or CUDA_ is used for fast number-crunching.  SOLVCON is designed
for extension to various physical processes.  Numerical algorithms and physical
models are pluggable.  Sub-package ``solvcon.kerpak`` contains default
implementations.  The default numerical algorithm in SOLVCON is the space-time
Conservation Element and Solution Element (CESE_) method, which was originally
developed by Sin-Chung Chang at NASA Glenn Research Center.  The CESE_ method
solves generic, first-order, hyperbolic PDEs.

SOLVCON is released under `GNU GPLv2
<http://www.gnu.org/licenses/gpl-2.0.html>`_, and developed by `Yung-Yu Chen
<mailto:yyc@solvcon.net>`_ and `Sheng-Tao John Yu <mailto:yu.274@osu.edu>`_.

Key Features
============

- **Multi-physics**: Pluggable physical models by the built-in CESE_ solvers
- **Complex geometry**: 2/3D unstructured mesh consisting of mixed shapes
- **Massively parallel**: Automatic domain decomposition with MPI or socket
- **GPGPU computing**: Hybrid parallelism with CUDA_
- **Large data set**: In situ visualization by VTK_ and parallel I/O
- **I/O formats**: VTK, GAMBIT Neutral, CUBIT Genesis/ExodosII, etc.
- **Productive work flow**: Integration to batch systems, e.g., Torque

Install
=======

The C code in SOLVCON are intentionally made to be standard shared libraries
rather than Python extension modules.  SOLVCON uses ctypes_ to load and call
these binary codes.  In this way, the binary codes can be flexibly built and
optimized for performance.  Hence, installing SOLVCON requires building these
libraries.  SOLVCON uses SCons_ as the binary builder.

For SOLVCON to be built and run, it requires the following packages: (i)
Python_ 2.6 or 2.7, (ii) SCons_, (iii) a C compiler, gcc_ or icc is OK, (iv)
Numpy_, (v) LAPACK_, (vi) NetCDF_ higher than version 4, and (vii) METIS_
version 4.0.3 for graph partitioning (SOLVCON will download it for you on
building).  Optional dependencies include: (i) SCOTCH_ (version 5.1 or higher)
as an alternative of METIS, (ii) Nose_ for running unit tests, (iii) Epydoc_
for generating API documentation, and (iv) VTK_ for in situ visualization.
64-bits Linux is recommended.  For Debian_ or Ubuntu_ users, they can use the
following command to install the dependencies::

  $ sudo apt-get install scons build-essential gcc liblapack-pic
    libnetcdf-dev libnetcdf6 netcdf-bin
    python2.6 python2.6-dev python-profiler python-numpy
    libscotch-5.1 python-nose python-epydoc python-vtk

CUDA_ needs to be separately installed and configured.  For using meshes with
more then 35 million cells, SCOTCH-5.1 is recommended.  METIS-4 has issues on
memory allocation for large graphs.

The end of this section describes how to manually compile these dependencies
with helper scripts shipped with SOLVCON.

The three steps to install:

1. Obtain the latest release from
   https://bitbucket.org/yungyuc/solvcon/downloads .  Unpack the source
   tarball.

2. Get into the source tree and run SCons_ to build the binary codes::

     $ cd $SCSRC
     $ scons --download --extract

   ``$SCSRC`` indicates the root directory of unpacked source tree.

3. Install everything::

     $ python setup.py install

   Optionally, you can install SOLVCON to your home directory.  It is useful
   when you don't have the root permission on the system.  To do this, add the
   ``--user`` when invoking the ``setup.py`` script::

     $ python setup.py install --user

The option ``--download`` used above asks the building script to download
necessary external source packages, e.g., METIS_, from Internet.  Option
``--extract`` extracts the downloaded packages.

If one wants to build the dependencies from ground up, he/she can take a look
into ``$SCSRC/ground``.  The directory contains scripts to compile most of the
depended packages, excepting very fundamental ones.  For those who like to get
their hands dirty, the following packages still need to be installed as
prerequisite::

  $ sudo apt-get install build-essential gcc cmake libcurl4-gnutls-dev
  libhdf5-serial-dev

Sometimes it is inevitable to compile these dependencies from source.  For
example, to deploy SOLVCON on a supercomputer/cluster which runs stable but
out-dated OSes.  The ``ground/`` directory is meant to ease the tedious task.

Install from Repository
=======================

To use the latest source from the code repository, you need to use Mercurial_
to clone the repository to your local disk::

  $ sudo apt-get install mercurial
  $ hg clone https://bitbucket.org/yungyuc/solvcon

and then follow steps 2 and 3.

Rebuild/Reinstall
=================

If you want to rebuild and reinstall, you can run::

  $ cd $SCSRC
  $ scons
  $ python setup.py install

without using the options ``--download`` and ``--extract``.  If you want a
clean rebuild, run ``scons -c`` before ``scons``.

Unit Test
=========

If you have Nose_ installed, you can run::

  $ nosetests

inside the source tree for unit tests.  To test installed version, use the
following command instead::

  $ python -c 'import solvcon; solvcon.test()'

When testing installed version, make sure your current directory does not have
a sub-directory named as ``solvcon``.

Because SOLVCON uses ssh_ as its default approach for remote procedure call
(RPC), you need to set up the public key authentication for ssh, or some of the
unit tests for RPC could fail.  Some tests using optional libraries could be
skipped (indicated by S), if you do not have the libraries installed.
Everything else should pass.

Resources
=========

- Portal (with API document): http://solvcon.net/
- Mailing list: http://groups.google.com/group/solvcon
- Downloads: http://bitbucket.org/yungyuc/solvcon/downloads

.. _CESE: http://www.grc.nasa.gov/WWW/microbus/
.. _SCons: http://www.scons.org/
.. _Python: http://www.python.org/
.. _gcc: http://gcc.gnu.org/
.. _Numpy: http://www.numpy.org/
.. _LAPACK: http://www.netlib.org/lapack/
.. _NetCDF: http://www.unidata.ucar.edu/software/netcdf/index.html
.. _METIS: http://glaros.dtc.umn.edu/gkhome/views/metis/
.. _SCOTCH: http://www.labri.fr/perso/pelegrin/scotch/
.. _Epydoc: http://epydoc.sf.net/
.. _CUDA: http://www.nvidia.com/object/cuda_home_new.html
.. _Mercurial: http://mercurial.selenic.com/
.. _ssh: http://www.openssh.com/
.. _Nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _VTK: http://vtk.org/
.. _ctypes: http://docs.python.org/library/ctypes.html
.. _Debian: http://debian.org/
.. _Ubuntu: http://ubuntu.com/
"""

__docformat__ = 'restructuredtext en'

__version__ = '0.0.6+'

__description__ = "SOLVCON: a software framework for PDE solvers"

__all__ = ['batch', 'batch_torque', 'block', 'boundcond', 'case',
    'cmdutil', 'command', 'conf', 'connection', 'dependency', 'domain',
    'gendata', 'helper', 'io', 'kerpak', 'mpy', 'mthread', 'rpc', 'scuda',
    'solver', 'visual_vtk']

from .cmdutil import go, test

if __name__ == '__main__':
    go()
