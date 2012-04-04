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
.. py:data:: env

  The global singleton of :py:class:`Solvcon` to provide configuration for
  other components of SOLVCON.
"""

__docformat__ = 'restructuredtext en'

class Solvcon(object):
    """
    The configuration class.

    :ivar pydir: The path of the :py:mod:`solvcon` package that is running.
    :type pydir: str
    :ivar pkgdir: The path that contains :py:mod:`solvcon` package that is
        running.
    :type pkgdir: str
    :ivar libdir: The path of the compiled binary of SOLVCON.
    :type libdir: str
    :ivar datadir: The path of the static data of SOLVCON.
    :type datadir: str
    :ivar projdir: The path that hosts a SOLVCON project.
    :type projdir: str
    :ivar logfile: The stream that saves runtime output.
    :type logfile: file
    :ivar logfn: The (absolute) path of the logfile.
    :type logfn: str
    :ivar modnames: Names of SOLVCON applications.
    :type modnames: list of str
    :ivar command: Unknown
    :ivar mpi: The MPI runtime interface.
    :type mpi: solvcon.mpy.MPI
    :ivar scu: The CUDA runtime interface.
    :type scu: solvcon.scuda.Scuda.
    """
    def __init__(self):
        import os, sys
        from ConfigParser import ConfigParser
        from .mpy import MPI
        from .scuda import Scuda
        # directories.
        self.pydir = os.path.abspath(os.path.dirname(__file__))
        self.pkgdir = os.path.abspath(os.path.join(self.pydir, '..'))
        ## library directory.
        paths = [os.path.join(self.pydir, '..', 'lib')] + [
            os.path.join(root, 'lib', 'solvcon') for root in (
                os.path.join(os.environ['HOME'], '.local'), sys.prefix)]
        self.libdir = self._get_first_existing_path(paths)
        ## data directory.
        paths = [os.path.join(self.pydir, '..', 'test', 'data')] + [
            os.path.join(root, 'share', 'solvcon', 'test') for root in (
                os.path.join(os.environ['HOME'], '.local'), sys.prefix)]
        self.datadir = self._get_first_existing_path(paths)
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
        self.modnames = [name.strip() for name in self.modnames if name]
        # data types.
        self._fpdtype = None
        # dynamic properties.
        self.command = None
        # MPI.
        self.mpi = os.environ.get('SOLVCON_MPI', None)
        self.mpi = MPI() if self.mpi is not None else self.mpi
        # CUDA.
        self.scu = Scuda() if Scuda.has_cuda() else None

    @staticmethod
    def _get_first_existing_path(paths):
        """
        Return the first existing path and turn it into an absolute path.

        :param paths: The list of paths.
        :type paths: list
        :return: The first existing path.
        :rtype: str
        """
        import os
        for path in paths:
            if os.path.exists(path):
                return os.path.abspath(path)
        raise ValueError('None of %s exists' % str(paths))

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

    def find_scdata_mesh(self):
        """
        Find the mesh directory of the scdata from the current working
        directory all the way to the root.

        :return: The path to the supplemental SOLVCON mesh data.
        :rtype: str
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

        :return: The invoked name.
        :rtype: str
        """
        import sys, os
        name = os.path.abspath(sys.argv[0])
        if name.find(sys.prefix) != -1 and name.find('scg') != -1:
            name = os.path.basename(name)
        return name

    def _enable_applications(self):
        """
        Enable a SOLVCON application by importing the module (or package).

        :return: Nothing.
        """
        for modname in self.modnames:
            if modname:
                raise ValueError("modname can't be '%s' (%s)" % (
                    modname, str(self.modnames)))
            else:
                __import__(modname, fromlist=['arrangement',])

env = Solvcon()
env._enable_applications()
