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
Information about the configuration of solvcon.

SOLVCON will find each of the solvcon.ini files from current working directory
toward the root, and use their settings.  Three settings are recognized in
[SOLVCON] section:

* APPS: SOLVCON_APPS
* LOGFILE: SOLVCON_LOGFILE
* PROJECT_DIR: SOLVCON_PROJECT_DIR.  Can be set at empty, which indicates the
    path where the configuration file locates.

Configurable environmental variables:

* ``SOLVCON_PROJECT_DIR``: the directory holds the applications.
* ``SOLVCON_LOGFILE``: filename for solvcon logfile.
* ``SOLVCON_APPS``: names of the available applications, seperated with
  semi-colon.  There should be no spaces.
* ``SOLVCON_FPDTYPE``: a string for the numpy dtype object for floating-point.
  The default fpdtype to be used is float64 (double).
* ``SOLVCON_INTDTYPE``: a string for the numpy dtype object for integer.
  The default intdtype to be used is int32.
* ``SOLVCON_FORTRAN``: flag to use FORTRAN binaries.
* ``SOLVCON_MPI``: flag to use MPI.
"""

__docformat__ = 'restructuredtext en'

class Solvcon(object):
    """
    The configuration singleton.
    """
    def __init__(self):
        import os, sys
        from ConfigParser import ConfigParser
        from .mpy import MPI
        from .scuda import Scuda
        # directories.
        self.pydir = os.path.abspath(os.path.dirname(__file__))
        self.pkgdir = os.path.abspath(os.path.join(self.pydir, '..'))
        libdir = os.path.join(self.pydir, '..', 'lib')
        for root in [sys.prefix, os.path.join(os.environ['HOME'], '.local')]:
            if os.path.exists(libdir):
                break
            else:
                libdir = os.path.join(root, 'lib', 'solvcon')
        self.libdir = os.path.abspath(libdir)
        datadir = os.path.join(self.pydir, '..', 'test', 'data')
        for root in [sys.prefix, os.path.join(os.environ['HOME'], '.local')]:
            if os.path.exists(datadir):
                break
            else:
                datadir = os.path.join(root, 'share', 'solvcon', 'test')
        self.datadir = os.path.abspath(datadir)
        # configuration files.
        cfgdirs = list()
        cwd = os.getcwd()
        parent = os.path.dirname(cwd)
        while parent != cwd:
            if os.path.exists(os.path.join(cwd, 'solvcon.ini')):
                cfgdirs.append(cwd)
            cwd = parent
            parent = os.path.dirname(cwd)
        cfg = ConfigParser()
        cfg.read([
            os.path.join(dname, 'solvcon.ini') for dname in cfgdirs])
        ## project directory.
        project_dir = None
        if cfg.has_option('SOLVCON', 'PROJECT_DIR'):
            project_dir = cfg.get('SOLVCON', 'PROJECT_DIR')
        if project_dir == '' and len(cfgdirs):
            project_dir = cfgdirs[0]
        projdir = os.environ.get('SOLVCON_PROJECT_DIR', project_dir)
        if projdir == None:
            projdir = os.getcwd()
        self.projdir = os.path.abspath(projdir)
        sys.path.insert(0, self.projdir)
        # logging.
        logfn = None
        if cfg.has_option('SOLVCON', 'LOGFILE'):
            logfn = cfg.get('SOLVCON', 'LOGFILE')
        logfn = os.environ.get('SOLVCON_LOGFILE', logfn)
        self.logfile = None if logfn == None else open(logfn, 'w')
        self.logfn = logfn
        # settings.
        if cfg.has_option('SOLVCON', 'APPS'):
            self.modnames = cfg.get('SOLVCON', 'APPS').split(';')
        else:
            self.modnames = list()
        emodnames = os.environ.get('SOLVCON_APPS', '').split(';')
        if emodnames:
            self.modnames.extend(emodnames)
        # data types.
        self._fpdtype = None
        self._intdtype = None
        # dynamic properties.
        self.command = None
        # library.
        self.use_fortran = bool(os.environ.get('SOLVCON_FORTRAN', False))
        # MPI.
        self.mpi = os.environ.get('SOLVCON_MPI', None)
        self.mpi = MPI() if self.mpi is not None else self.mpi
        # CUDA.
        self.scu = Scuda() if Scuda.has_cuda() else None

    @property
    def fpdtype(self):
        import os
        import numpy as np
        if self._fpdtype != None:
            _fpdtype = self._fpdtype
        else:
            dtypestr = os.environ.get('SOLVCON_FPDTYPE', 'float64')
            _fpdtype = getattr(np, dtypestr)
            self._fpdtype = _fpdtype
        return _fpdtype

    @property
    def fpdtypestr(self):
        import numpy as np
        for dtypestr in 'float64', 'float32':
            if self.fpdtype == getattr(np, dtypestr):
                return dtypestr

    @property
    def intdtype(self):
        """
        INACTIVE; PLANNED FOR FUTURE USE.
        """
        import os
        import numpy as np
        if self._intdtype != None:
            _intdtype = self._intdtype
        else:
            dtypestr = os.environ.get('SOLVCON_INTDTYPE', 'int32')
            _intdtype = getattr(np, dtypestr)
            self._intdtype = _intdtype
        return _intdtype

    def find_scdata_mesh(self):
        """
        Find the mesh directory of the scdata from the current working
        directory all the way to the root.
        """
        import os
        from .helper import search_in_parents
        scdata = search_in_parents(os.getcwd(), 'scdata')
        if not scdata:
            raise OSError('cannot find scdata directory')
        return os.path.join(scdata, 'mesh')

    def get_entry_point(self):
        """
        If the entry point is invoked by searching in path, just return it.  If
        the entry point is invoked by specifying a location, return the
        absolute path of the entry script/code.
        """
        import sys, os
        name = os.path.abspath(sys.argv[0])
        if name.find(sys.prefix) != -1 and name.find('scg') != -1:
            name = os.path.basename(name)
        return name

env = Solvcon()

def use_application(modname):
    if modname:
        __import__(modname, fromlist=['arrangement',])
