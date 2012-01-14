# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2012 Yung-Yu Chen <yyc@solvcon.net>.
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
SOLVCON: a multi-physics software framework for high-fidelity solutions of
partial differential equations (PDEs) by hybrid parallelism.

SOLVCON uses the space-time Conservation Element and Solution Element (`CESE
<http://www.grc.nasa.gov/WWW/microbus/>`_) method to solve generic conservation
laws.  SOLVCON focuses on rapid development of high-performance computing (HPC)
code for large-scale simulations.  SOLVCON is developed by using Python for the
main structure, to incorporate C, `CUDA
<http://www.nvidia.com/object/cuda_home_new.html>`_, or other programming
languages for HPC.

SOLVCON is released under `GNU GPLv2
<http://www.gnu.org/licenses/gpl-2.0.html>`_, and developed by `Yung-Yu Chen
<mailto:yyc@solvcon.net>`_ and `Sheng-Tao John Yu <mailto:yu.274@osu.edu>`_.
The official web site is at http://solvcon.net/ .

Key Features
============

- Pluggable multi-physics
- Unstructured meshes for modeling complex geometry
- Hybrid parallel computing
- Ready-to-use I/O formats
- Parallel I/O and in situ visualization
- Automated work flow

Prerequisite
============

SOLVCON depends on the following packages:

- `gcc <http://gcc.gnu.org/>`_ 4.3+
- `SCons <http://www.scons.org/>`_ 2+
- `Python <http://www.python.org/>`_ 2.6 or 2.7 (2.7 is preferred)
- `Numpy <http://www.numpy.org/>`_ 1.5+
- `LAPACK <http://www.netlib.org/lapack/>`_
- `NetCDF <http://www.unidata.ucar.edu/software/netcdf/index.html>`_ 4+
- `METIS <http://glaros.dtc.umn.edu/gkhome/views/metis/>`_ 4.0.3+ (SOLVCON will
  download it for you on building) or `SCOTCH
  <http://www.labri.fr/perso/pelegrin/scotch/>`_ 5.1+
- `Nose <http://somethingaboutorange.com/mrl/projects/nose/>`_ 1.0+
- `gmsh <http://geuz.org/gmsh/>`_ 2.5+
- `VTK <http://vtk.org/>`_ 5.6+

The following command will install these packages on Debian/Ubunbu::

  $ sudo apt-get install build-essential gcc scons liblapack-pic libnetcdf-dev
  libnetcdf6 netcdf-bin libscotch-5.1 python2.7 python2.7-dev python-numpy
  python-nose gmsh python-vtk

Note: For Debian Squeeze (6.0), you need also ``apt-get install
python-profiler`` to install the built-in Python profiler.

Building document of SOLVCON requires the following packages:

- `Epydoc <http://epydoc.sf.net/>`_ 3+
- `pygraphviz <http://networkx.lanl.gov/pygraphviz/>`_ 1.1+
- `Sphinx <http://sphinx.pocoo.org/>`_ 1.0+

which can be installed on Debian/Ubunbu by using the command::

  $ sudo apt-get install python-epydoc python-pygraphviz python-sphinx

Another optional dependency is CUDA_, which needs to be separately installed
and configured.  For using meshes with more then 35 million cells, SCOTCH-5.1
is recommended.  METIS-4 has issues on memory allocation for large graphs.

Install
=======

There are three steps to install SOLVCON:

1. Obtain the latest release from
   https://bitbucket.org/yungyuc/solvcon/downloads .  Unpack the source
   tarball.

2. Get into the source tree and run SCons_ to build the binary codes::

     $ cd $SCSRC
     $ scons --download --extract

   where ``$SCSRC`` indicates the root directory of unpacked source tree.

3. Install everything::

     $ python setup.py install

The option ``--download`` used above lets the building script download
necessary external source packages, e.g., METIS_, from Internet.  Option
``--extract`` extracts the downloaded packages.

Although not recommended, you can optionally install SOLVCON to your
``$HOME/.local`` directory.  It is one of the workarounds when you don't have
the root permission on the system.  To do this, add the ``--user`` when
invoking the ``setup.py`` script::

 $ python setup.py install --user

SOLVCON is designed to work without explicit installation.  You can simply set
the ``$PYTHONPATH`` environment variable to point to the unpacked source
distribution (``$SCSRC``).  Compilation of binary code by using SCons is still
required.

Development Version
===================

To use the latest development version, you need to use `Mercurial
<http://mercurial.selenic.com/>`_ to access the source repository.  Clone the
repository::

  $ sudo apt-get install mercurial
  $ hg clone https://bitbucket.org/yungyuc/solvcon

and follow steps 2 and 3 in Install_.

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

inside the source tree for unit tests.  To test the installed package, use the
following command instead::

  $ python -c 'import solvcon; solvcon.test()'

When testing the installed package, make sure your current directory does not
have a sub-directory named as ``solvcon``.

Because SOLVCON uses `ssh <http://www.openssh.com/>`_ as its default approach
for remote procedure call (RPC), you need to set up the public key
authentication for ssh, or some of the unit tests for RPC could fail.  Some
tests using optional libraries could be skipped (indicated by S), if you do not
have the libraries installed.  Everything else should pass.

Build and Install Dependencies (Optional)
=========================================

SOLVCON depends on a number of external software packages.  Although these
dependencies should be taken care by OSes, it takes time to get the support
personnels to install missing packages on a cluster/supercomputer.  As such,
SOLVCON provides a simple building system to facilitate the installation into a
customizable location.

The ``$SCSRC/ground`` directory contains scripts to build most of the packages
that SOLVCON depends on.  The ``$SCSRC/ground/get`` script downloads the source
packages to be built.  The ``$SCSRC/ground/Makefile`` file has three default
targets: ``binary``, ``python``, and ``vtk``.  The built files will be
automatically installed into the path specified by the ``$SCROOT`` environment
variable, which is set to ``$HOME/opt/scruntime`` by default.  The
``$SCROOT/bin/scvars.sh`` script will be created to export necessary
environment variables for the installed software, and the ``$SCROOT``
environment variable itself.

The ``$SCSRC/soil`` directory contains scripts to build gcc_.  The
``$SCROOT/bin/scgccvars.sh`` script will be created to export necessary
environment variables for the self-compiled gcc.  The enabled languages include
only C, C++, and Fortran.  The default value of ``$SCROOT`` remains to be
``$HOME/opt/scruntime``, while the software will be installed into
``$SCROOT/soil``.  Note: (i) Do not use different ``$SCROOT`` when building
``$SCSRC/soil`` and ``$SCSRC/ground``.  (ii) On hyper-threading CPUs the ``NP``
environment variable should be set to the actual number of cores, or
compilation of gcc could exhaust system memory.

``$SCROOT/bin/scvars.sh`` and ``$SCROOT/bin/scgccvars.sh`` can be separately
sourced.  The two sets of packages reside in different directories and do not
mix with each other nor system software.  Users can disable these environments
by not sourcing the two scripts.

Some packages have not been incorporated into the dependency building system
described above.  Debian or Ubuntu users should install the additional
dependencies by using::

  $ sudo apt-get install build-essential gcc gfortran gcc-multilib m4
   libreadline6 libreadline6-dev libncursesw5 libncurses5-dev libbz2-1.0
   libbz2-dev libdb4.8 libdb-dev libgdbm3 libgdbm-dev libsqlite3-0
   libsqlite3-dev libcurl4-gnutls-dev libhdf5-serial-dev libgl1-mesa-dev
   libxt-dev

These building scripts have only been tested with 64-bit Linux.
"""

__docformat__ = 'restructuredtext en'

__version__ = '0.1+'

__description__ = "SOLVCON: a software framework for PDE solvers"

__all__ = ['batch', 'batch_torque', 'block', 'boundcond', 'case',
    'cmdutil', 'command', 'conf', 'connection', 'dependency', 'domain',
    'gendata', 'helper', 'io', 'kerpak', 'mpy', 'mthread', 'rpc', 'scuda',
    'solver', 'visual_vtk']

from .cmdutil import go, test

if __name__ == '__main__':
    go()
