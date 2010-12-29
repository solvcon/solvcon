# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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

Python is the primary programming language for constructing SOLVCON.
Number-crunching is performed by high-speed subroutines written in C.  By
taking the mixed-language approach, PDE solvers can be rapidly developed and
simultaneously utilize hundreds of nodes in a supercomputer by parallel
computing.  SOLVCON is multi-physics, and stocking numerical algorithms and
physical models are ready for use in the namespace ``solvcon.kerpak``.

The default numerical algorithm in SOLVCON is the space-time Conservation
Element and Solution Element (CESE_) method, which was originally developed by
Sin-Chung Chang at NASA Glenn Research Center.  The CESE_ method delivers
time-accurate solutions of hyperbolic PDEs, and has been used to solve various
physical processes including fluid dynamics, aero-acoustics, detonations,
magnetohydrodynamics (MHD), stress waves in complex solids, electromagnetics,
to be named but a few.

SOLVCON is free software (for freedom, not price) and released under GPLv2.
See http://www.gnu.org/licenses/gpl-2.0.html or ``COPYING`` for the complete
license.  **SOLVCON is still in alpha and subjects to changes.  No effort is
made for backward compatibility at the current stage.**

Credits
=======

SOLVCON is developed by `Yung-Yu Chen <mailto:yyc@solvcon.net>`_ and `Sheng-Tao
John Yu <mailto:yu.274@osu.edu>`_.

Key Features
============

- Pluggable multi-physics.
- Built-in CESE_ solvers.
- Unstructured mesh consisting of mixed elements.
- Interface to Message-Passing Interface (MPI) libraries.
- Socket communication layer: working without MPI installed.
- Automatic distributed-memory parallelization by domain decomposition.
- Hybrid parallelism for GPU clusters.
- Parallel I/O.
- In situ visualization by VTK_ library.
- Standalone writers to VTK legacy and XML file formats.
- Integration to supercomputer (cluster) batch systems.

Install
=======

The C codes in SOLVCON are intentionally made to be generic shared libraries
rather than Python extension modules.  SOLVCON uses ``ctypes`` to load and call
these binary codes.  In this way, the binary codes can be flexibly built and
optimized for performance.  Hence, installing SOLVCON requires building these
libraries.  SOLVCON uses SCons_ as the binary builder.

For SOLVCON to be built and run, it requires the following packages: (i)
Python_ 2.6, (ii) SCons_, (iii) a C compiler, gcc_ or icc is OK, (iv) Numpy_,
(v) LAPACK, and (vi) METIS_ for graph partitioning (SOLVCON will download it
for you on building).  If you want to run the unit tests after building
SOLVCON, you should also install Nose_.  64-bit Linux is recommended.  For
Debian_ or Ubuntu_ users, you can use the following command to install the
dependency::

  $ sudo apt-get install python2.6 python2.6-dev python-profiler scons \\
    build-essential gcc python-numpy python-nose python-vtk liblapack-pic

Procedures to install are:

1. First, obtain the latest release from
   https://bitbucket.org/yungyuc/solvcon/downloads .  Unpack the source
   tarball.  Assume ``$SCSRC`` indicates the root directory of unpacked source
   tree.

2. Get into the source tree and run SCons_ to build the binary codes::

     $ cd $SCSRC
     $ scons --download --extract --apply-patches=metislog2

3. Install everything::

     $ python setup.py install

The option ``--download`` used above asks the building script to download
necessary external packages, e.g., METIS, from Internet.  Option ``--extract``
extracts the downloaded packages.  Since METIS is incompatible to the current
release of gcc, a patch is supplied with SOLVCON and can be automatically
applied to the downloaded METIS source with the ``--apply-patches`` option.

If you want to rebuild the binary after the installation, you can run::

  $ cd $SCSRC
  $ scons
  $ python setup.py install

without using the options ``--download``, ``--extract``, and
``--apply-patches``.  If you want a clean rebuild, run ``scons -c`` before
``scons``.

Test
====

SOLVCON uses ssh to bootstrap the remote procedure call for its socket
communication layer.  For tests to be run correctly, you must `have ssh public
key authentication configured
<http://www.google.com/search?q=ssh+public+key+authentication>`_.  If you
haven't done so, you can use the following commands to configure::

  $ ssh-keygen -t rsa -b 2048
  $ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
  $ chmod 400 ~/.ssh/authorized_keys

If you have Nose_ installed, you can run::

  $ python -c 'import solvcon; solvcon.test()'

for unit tests.  Every test should pass, except one specific to cluster batch
systems could be skipped (indicated by S).  If you do not have VTK_ and its
Python binding, VTK-related tests will also be skipped.

In Situ Visualization
=====================

Several pre-defined visualizing operations are built in SOLVCON by using VTK
library.  To use the provided in situ visualization, please make sure VTK and
its Python binding is installed correctly.

How to Use
==========

Examples for using SOLVCON are put in ``$SCSRC/examples``.  To run these
examples, you need corresponding mesh data, which is kept in a standalone
repository at https://bitbucket.org/yungyuc/scdata .  The ``scdata`` directory
should be downloaded in a directory higher than ``$SCSRC/examples``.  The
examples will find the ``scdata`` directory automatically.

These examples are useful for you to learn how to use SOLVCON to construct your
own solvers or applications.  Please read them in detail.

Resources
=========

- Portal: http://solvcon.net/
- Mailing list: http://groups.google.com/group/solvcon
- Issue tracker (bug report): https://bitbucket.org/yungyuc/solvcon/issues
- Source: https://bitbucket.org/yungyuc/solvcon/src
- Downloads: https://bitbucket.org/yungyuc/solvcon/downloads

.. _CESE: http://www.grc.nasa.gov/WWW/microbus/
.. _SCons: http://www.scons.org/
.. _Python: http://www.python.org/
.. _gcc: http://gcc.gnu.org/
.. _Numpy: http://www.numpy.org/
.. _METIS: http://glaros.dtc.umn.edu/gkhome/views/metis/
.. _Nose: http://somethingaboutorange.com/mrl/projects/nose/
.. _VTK: http://vtk.org/
.. _Debian: http://debian.org/
.. _Ubuntu: http://ubuntu.com/
"""

__docformat__ = 'restructuredtext en'

__version__ = '0.0.1+'

__description__ = "Solver Constructor: a framework to solve hyperbolic PDEs"

__all__ = ['batch', 'batch_torque', 'block', 'boundcond', 'case',
    'cmdutil', 'command', 'conf', 'connection', 'dependency', 'domain',
    'gendata', 'helper', 'io', 'kerpak', 'mpy', 'mthread', 'rpc', 'solver',
    'visual_vtk']

from .cmdutil import go, test

if __name__ == '__main__':
    go()
