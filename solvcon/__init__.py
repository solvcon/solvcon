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
SOLVCON: a multi-physics software framework for high-fidelity solutions of
partial differential equations (PDEs) by hybrid parallelism.

SOLVCON primarily uses Python_, but number-crunching is done by C subroutines
for speed.  The mixing-language approach allows PDE solvers to be rapidly
developed for massively parallel computing.  The object-oriented design
facilitated by Python makes SOLVCON easy to be systematically extended.
SOLVCON supports pluggable physical models.  Stocking ones are accessible in
the namespace ``solvcon.kerpak``.

The default numerical algorithm in SOLVCON is the space-time Conservation
Element and Solution Element (CESE_) method, which was originally developed by
Sin-Chung Chang at NASA Glenn Research Center.  The CESE_ method delivers
time-accurate solutions of hyperbolic PDEs, and has been used to solve various
physical processes including fluid dynamics, aero-acoustics, detonations,
magnetohydrodynamics (MHD), stress waves in complex solids, electromagnetics,
to be named but a few.

SOLVCON is free software and released under GPLv2.  See
http://www.gnu.org/licenses/gpl-2.0.html or ``COPYING`` for the license.
SOLVCON is being developed and maintained by `Yung-Yu Chen
<mailto:yyc@solvcon.net>`_ and `Sheng-Tao John Yu <mailto:yu.274@osu.edu>`_.

Key Features
============

- **Multi-physics**: Pluggable physical models by the built-in CESE_ solvers
- **Complex geometry**: 2/3D unstructured mesh consisting of mixed shapes
- **Massively parallel**: Automatic domain decomposition with MPI or socket
- **GPU cluster**: Hybrid parallelism with CUDA_
- **Large data set**: In situ visualization by VTK_ and parallel I/O
- **I/O formats**: VTK, GAMBIT Neutral, CUBIT Genesis/ExodosII, etc.
- **Productive work flow**: Integration to batch systems, e.g., Torque

Install
=======

The C codes in SOLVCON are intentionally made to be standard shared libraries
rather than Python extension modules.  SOLVCON uses ctypes_ to load and call
these binary codes.  In this way, the binary codes can be flexibly built and
optimized for performance.  Hence, installing SOLVCON requires building these
libraries.  SOLVCON uses SCons_ as the binary builder.

For SOLVCON to be built and run, it requires the following packages: (i)
Python_ 2.6, (ii) SCons_, (iii) a C compiler, gcc_ or icc is OK, (iv) Numpy_,
(v) LAPACK_, (vi) NetCDF_ higher than version 4, and (vii) METIS_ version 4 for
graph partitioning (SOLVCON will download it for you on building).  Optional
dependencies include: (i) SCOTCH_ (higher than version 5.1) as an alternative
of METIS, (ii) Nose_ for running unit tests, (iii) Epydoc_ for generating API
documentation, and (iv) VTK_ for in situ visualization.  64-bits Linux is
recommended.  For Debian_ or Ubuntu_ users, they can use the following command
to install the dependencies::

  $ sudo apt-get install scons build-essential gcc liblapack-pic
    libnetcdf-dev libnetcdf6 netcdf-bin
    python2.6 python2.6-dev python-profiler python-numpy
    libscotch-5.1 python-nose python-epydoc python-vtk

CUDA_ needs to be separately installed and configured for the CUDA support in
SOLVCON to correctly work.  For dealing with meshes with more then 35 million
cells, SCOTCH-5.1 is recommended, for METIS-4 has issues on memory allocation.

Installation needs only three steps:

1. First, obtain the latest release from
   https://bitbucket.org/yungyuc/solvcon/downloads .  Unpack the source
   tarball.  Let ``$SCSRC`` indicate the root directory of unpacked source
   tree.

2. Get into the source tree and run SCons_ to build the binary codes::

     $ cd $SCSRC
     $ scons --download --extract --apply-patches=metislog2

3. Install everything::

     $ python setup.py install

The option ``--download`` used above asks the building script to download
necessary external source packages, e.g., METIS_, from Internet.  Option
``--extract`` extracts the downloaded packages.  Since METIS is incompatible to
the current release of gcc, a patch is supplied with SOLVCON and can be
automatically applied to the downloaded METIS source with the
``--apply-patches`` option.

Install from Repository
=======================

Since SOLVCON is in intensive development, you may want to use the latest
source from the code repository.  You need to install Mercurial_, clone the
repository to your local disk::

  $ sudo apt-get install mercurial
  $ hg clone https://bitbucket.org/yungyuc/solvcon

and then follow steps 2 and 3.

Rebuild/Reinstall
=================

If you want to rebuild and reinstall, you can run::

  $ cd $SCSRC
  $ scons
  $ python setup.py install

without using the options ``--download``, ``--extract``, and
``--apply-patches``.  If you want a clean rebuild, run ``scons -c`` before
``scons``.  Note, ``scons -c`` does not remove the unpacked source, so you
don't need to reapply the patches unless you have deleted the unpacked source
files.

You can optionally install SOLVCON to your home directory rather than to the
system.  It is convenient when you don't have the root permission on the
system.  To do that, add the ``--user`` when invoking the ``setup.py`` script::

  $ python setup.py install --user

Unit Test
=========

If you have Nose_ installed, you can run::

  $ python -c 'import solvcon; solvcon.test()'

for unit tests.  Inside the source tree, you can also use::

  $ nosetests

Because SOLVCON uses ssh_ as its default approach for remote procedure call
(RPC), you need to set up the public key authentication for ssh, or some of the
unit tests for RPC could fail.  Every test should pass, except some
non-applicable to your environment could be skipped (indicated by S).  For
example, if you do not have VTK_ and its Python binding, VTK-related tests will
be skipped.

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

__version__ = '0.0.2+'

__description__ = "SOLVCON: a software framework for PDE solvers"

__all__ = ['batch', 'batch_torque', 'block', 'boundcond', 'case',
    'cmdutil', 'command', 'conf', 'connection', 'dependency', 'domain',
    'gendata', 'helper', 'io', 'kerpak', 'mpy', 'mthread', 'rpc', 'scuda',
    'solver', 'visual_vtk']

from .cmdutil import go, test

if __name__ == '__main__':
    go()
