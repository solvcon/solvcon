# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Access Torque.
"""

from ctypes import Structure
class TmRoots(Structure):
    """
    The data structure for torque tm API initialization.
    """
    from ctypes import c_uint, c_int, c_void_p
    _fields_ = [
        ('tm_me', c_uint),
        ('tm_parent', c_uint),
        ('tm_nnodes', c_int),
        ('tm_ntasks', c_int),
        ('tm_taskpoolid', c_int),
        ('tm_tasklist', c_void_p),
    ]
del Structure

class TaskManager(object):
    """
    Torque task manager API wrapper.

    @cvar _clib_torque: the DLL for Torque.
    @ctype _clib_torque: ctypes.dll
    @cvar tmroots: the tm_roots structure for TM API initialization.
    @ctype tmroots: TmRoots

    @ivar nodelist: the list of node ids.
    @itype nodelist: list
    """

    import os
    from ctypes import c_void_p, byref, CDLL
    # load the library if can find it.
    _clib_torque = CDLL(os.path.join(os.environ['PBS_INSTALL'],
        'lib', 'libtorque.so')) if 'PBS_INSTALL' in os.environ else None
    # initialize torque library.  NOTE: it can be initialized only once.
    tmroots = TmRoots()
    if _clib_torque:
        _clib_torque.tm_init(c_void_p(), byref(tmroots))
    del c_void_p, byref, CDLL, os

    def __init__(self, paths=None):
        from .conf import env
        # list of node ids.
        self.nodelist = self._get_nodelist() if self._clib_torque else []
        # paths.
        paths = {} if paths == None else paths.copy()
        if env.pkgdir not in paths['PYTHONPATH']:
            paths['PYTHONPATH'].append(env.pkgdir)
        for key in paths:
            paths[key] = self._pathmunge(key, paths[key])
        self.paths = paths

    @staticmethod
    def _pathmunge(key, pathlist):
        from .helper import iswin
        pathlist = pathlist[:]
        # scan for duplication.
        ip = 0
        while ip < len(pathlist):
            jp = ip + 1
            while jp < len(pathlist):
                if pathlist[ip] == pathlist[jp]:
                    del pathlist[jp]
                else:
                    jp += 1
            ip += 1
        # join the path.
        sep = ';' if iswin() else ':'
        pathstr = sep.join([path for path in pathlist if path])
        return '%s:$%s' % (pathstr, key)

    def _get_nodelist(self):
        """
        Get nodelist by using TM API.

        @return: the list of node id (integer).
        @rtype: list
        """
        from ctypes import c_int, POINTER, byref
        nodelist = POINTER(c_int)()
        nnode = c_int()
        self._clib_torque.tm_nodeinfo(byref(nodelist), byref(nnode))
        assert nnode.value == self.tmroots.tm_nnodes
        return [nodelist[it] for it in range(nnode.value)]

    def spawn(self, *args, **kw):
        """
        Spawn a process on designated node with given location and
        environment variables.

        @keyword where: which node to run the spawned process.
        @type where: int
        @keyword envar: environmental variables.
        @type envar: dict
        @return: tid and event.
        @rtype: tuple
        """
        import os
        from ctypes import c_void_p, c_int, c_char_p, byref
        # get where.
        where = kw.pop('where')
        # get environment variables.
        envar = self.paths.copy()
        envar.pop('LD_LIBRARY_PATH', None)  # the var causes segfault.
        envar['PBS_INSTALL'] = os.environ['PBS_INSTALL']
        envar.update(kw.pop('envar', dict()))
        envp = byref((c_char_p*len(envar))(*[
            '%s=%s' % (k, envar[k]) for k in envar]))
        # spawn.
        tid = c_int()
        event = c_int()
        self._clib_torque.tm_spawn(
            c_int(len(args)),
            byref((c_char_p*len(args))(*args)),
            envp,
            c_int(where),
            byref(tid),
            byref(event),
        )
        return tid.value, event.value

def run_worker(hostaddr, cwd, profiler_data, inetaddr, authkey):
    """
    Run a worker on slave node.
    """
    import os, sys
    from solvcon.connection import pick_unused_port, Client
    from solvcon.rpc import Worker
    # guess and report port.
    port = pick_unused_port()
    conn = Client(hostaddr, authkey=authkey)
    conn.send(port)
    conn.close()
    # run worker object.
    os.chdir(cwd)
    wkr = Worker(None, profiler_data=profiler_data)
    wkr.run((inetaddr, port), authkey)
