# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Information about the configuration of solvcon.

Configurable environmental variables:

* ``SOLVCON_PROJECT_DIR``: the directory holds the applications.
* ``SOLVCON_LOGFILE``: filename for solvcon logfile.
* ``SOLVCON_APPS``: names of the available applications, seperated with
  semi-colon.  There should be no spaces.
* ``SOLVCON_FPDTYPE``: a string for the numpy dtype object for floating-point.
  The default fpdtype to be used is float64 (double).
* ``SOLVCON_FORTRAN``: flag to use FORTRAN binaries.
* ``SOLVCON_MPI``: flag to use MPI.
"""

__docformat__ = 'restructuredtext en'

class Solvcon(object):
    """
    The configuration singleton.
    """
    def __init__(self):
        import os
        from .mpy import MPI
        # directories.
        self.pydir = os.path.abspath(os.path.dirname(__file__))
        self.pkgdir = os.path.abspath(os.path.join(self.pydir, '..'))
        self.libdir = os.path.abspath(os.path.join(self.pydir, '..', 'lib'))
        self.datadir = os.path.abspath(
            os.path.join(self.pydir, '..', 'test', 'data'))
        ## project directory.
        projdir = os.environ.get('SOLVCON_PROJECT_DIR', None)
        if projdir == None:
            projdir = os.getcwd()
        self.projdir = os.path.abspath(projdir)
        # logging.
        logfn = os.environ.get('SOLVCON_LOGFILE', None)
        self.logfile = None if logfn == None else open(logfn, 'w')
        self.logfn = logfn
        # settings.
        self.modnames = os.environ.get('SOLVCON_APPS', '').split(';')
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

env = Solvcon()

def use_application(modname):
    __import__(modname, fromlist=['arrangement',])
