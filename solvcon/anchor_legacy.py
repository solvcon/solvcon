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
Anchors attached to the solvers.  There's only one base anchor class for
subclassing.  Any other anchors defined here are for directly installation.

A special GlueAnchor is defined to glue two collocated BCs.  The pair of glued
BCs works as an internal interface.  As such, the BCs can be dynamically turn
on or off.
"""


import sys
import os
import time
import ctypes

import numpy as np

# execution.
from . import boundcond
# employment.
from .io import vtkxml
from . import visual_vtk


class Anchor(object):
    """
    Anchor that called by solver objects at various stages.

    @ivar svr: the solver object to be attached to.
    @itype svr: solvcon.solver.Solver
    @ivar kws: excessive keywords.
    @itype kws: dict
    """

    def __init__(self, svr, **kw):
        from . import solver # work around cyclic importation.
        assert isinstance(svr, (solver.BaseSolver, solver.MeshSolver))
        self.svr = svr
        self.kws = dict(kw)

    def provide(self):
        pass
    def preloop(self):
        pass
    def premarch(self):
        pass
    def prefull(self):
        pass
    def presub(self):
        pass
    def postsub(self):
        pass
    def postfull(self):
        pass
    def postmarch(self):
        pass
    def postloop(self):
        pass
    def exhaust(self):
        pass

class AnchorList(list):
    """
    Anchor container and invoker.

    @ivar svr: solver object.
    @itype svr: solvcon.solver.BaseSolver
    """
    def __init__(self, svr, *args, **kw):
        self.svr = svr
        self.names = dict()
        super(AnchorList, self).__init__(*args, **kw)
    def append(self, obj, **kw):
        name = kw.pop('name', None)
        if isinstance(name, int):
            raise ValueError('name can\'t be integer')
        if isinstance(obj, type):
            obj = obj(self.svr, **kw)
        super(AnchorList, self).append(obj)
        if name != None:
            self.names[name] = obj
    def __getitem__(self, key):
        if key in self.names:
            return self.names[key]
        else:
            return super(AnchorList, self).__getitem__(key)
    def __call__(self, method):
        """
        Invoke the specified method for each anchor.
        
        @param method: name of the method to run.
        @type method: str
        @return: nothing
        """
        runanchors = self.svr.runanchors
        if method == 'postloop' or method == 'exhaust':
            runanchors = reversed(runanchors)
        for anchor in runanchors:
            func = getattr(anchor, method, None)
            if func != None:
                func()

################################################################################
# Solution output.
################################################################################

class MarchSaveAnchor(Anchor):
    """
    Save solution data into VTK XML format for a solver.

    @ivar anames: the arrays in der of solvers to be saved.  True means in der.
    @itype anames: dict
    @ivar compressor: compressor for binary data.  Can only be 'gz' or ''.
    @itype compressor: str
    @ivar fpdtype: string for floating point data type (in numpy convention).
    @itype fpdtype: str
    @ivar psteps: the interval (in step) to save data.
    @itype psteps: int
    @ivar vtkfn_tmpl: the template string for the VTK file.
    @itype vtkfn_tmpl: str
    """
    def __init__(self, svr, **kw):
        self.anames = kw.pop('anames', dict())
        self.compressor = kw.pop('compressor')
        self.fpdtype = kw.pop('fpdtype')
        self.psteps = kw.pop('psteps')
        self.vtkfn_tmpl = kw.pop('vtkfn_tmpl')
        super(MarchSaveAnchor, self).__init__(svr, **kw)
    def _write(self, istep):
        from . import solver # work around for cyclic importation.
        ngstcell = self.svr.ngstcell
        sarrs = dict()
        varrs = dict()
        # collect data.
        for key in self.anames:
            # get the array.
            if self.anames[key]:
                arr = self.svr.der[key][ngstcell:]
            else:
                arr = getattr(self.svr, key)[ngstcell:]
            # put array in dict.
            if len(arr.shape) == 1:
                sarrs[key] = arr
            elif arr.shape[1] == self.svr.ndim:
                varrs[key] = arr
            else:
                for it in range(arr.shape[1]):
                    sarrs['%s[%d]' % (key, it)] = arr[:,it]
        # write.
        wtr = vtkxml.VtkXmlUstGridWriter(
            solver.FakeBlockVtk(self.svr), fpdtype=self.fpdtype,
            compressor=self.compressor, scalars=sarrs, vectors=varrs)
        svrn = self.svr.svrn
        wtr.write(self.vtkfn_tmpl % (istep if svrn is None else (istep, svrn)))
    def preloop(self):
        self._write(0)
    def postmarch(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps == 0:
            self._write(istep)
    def postloop(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps != 0:
            self._write(istep)

################################################################################
# Anchors for in situ visualization.
################################################################################

class VtkAnchor(Anchor):
    """
    Abstract class for VTK filtering anchor.  Must override process() method
    for use.  Note: svr.ust is shared by all VtkAnchor instances.

    @ivar anames: the arrays in der of solvers to be saved.  True means in der.
    @itype anames: dict
    @ivar fpdtype: string for floating point data type (in numpy convention).
    @itype fpdtype: str
    @ivar psteps: the interval (in step) to save data.
    @itype psteps: int
    @ivar vtkfn_tmpl: the template string for the VTK file.
    @itype vtkfn_tmpl: str
    """
    def __init__(self, svr, **kw):
        self.anames = kw.pop('anames', dict())
        self.fpdtype = kw.pop('fpdtype')
        self.psteps = kw.pop('psteps')
        self.vtkfn_tmpl = kw.pop('vtkfn_tmpl')
        self.vac = dict()
        super(VtkAnchor, self).__init__(svr, **kw)
    @property
    def vtkfn(self):
        """
        The correct file name for VTK based on the template.
        """
        istep = self.svr.step_global
        svrn = self.svr.svrn
        return self.vtkfn_tmpl % (
            istep if svrn is None else (istep, svrn))
    def _aggregate(self):
        """
        Aggregate data from solver object to VTK unstructured mesh.

        @return: nothing
        """
        ngstcell = self.svr.ngstcell
        fpdtype = self.fpdtype
        ust = self.svr.ust
        # collect derived.
        for key, inder, flag in self.anames:
            # get the array.
            if inder:
                arr = self.svr.der[key][ngstcell:]
            else:
                arr = getattr(self.svr, key)[ngstcell:]
            # set array in unstructured mesh.
            if len(arr.shape) == 1:
                visual_vtk.set_array(arr, key, fpdtype, ust)
            elif arr.shape[1] == self.svr.ndim:
                visual_vtk.set_array(
                    visual_vtk.valid_vector(arr), key, fpdtype, ust)
            else:
                for it in range(arr.shape[1]):
                    visual_vtk.set_array(
                        arr[:,it], '%s[%d]' % (key, it), fpdtype, ust)
    def preloop(self):
        self.process(0)
    def postmarch(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps == 0:
            self.process(istep)
    def postloop(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps != 0:
            self.process(istep)
    def process(self, istep):
        """
        This method implements the VTK filtering operations.  Must be
        overidden.
        """
        raise NotImplementedError

################################################################################
# StatAnchor.
################################################################################

class RuntimeStatAnchor(Anchor):
    """
    Report the Linux load average through solver.  Reports are made after a
    full marching interation.

    @ivar reports: list what should be reported.  Default is ['loadavg'] only.
    @itype reports: list
    @ivar cputotal: flag to use total jiffy for cpu usage percentage.
    @itype cputotal: bool
    @ivar cputime: marker for timing cpu usage.
    @itype cputime: float
    @ivar jiffytime: the time a jiffy is.  Default is 0.01 second.
    @itype jiffytime: float
    """

    PSTAT_KEYS = [
        ('pid', int), ('comm', str), ('state', str),
        ('ppid', int), ('pgrp', int), ('session', int),
        ('tty_nr', int), ('tpgid', int), ('flags', int),
        ('minflt', int), ('cminflt', int), ('majflt', int), ('cmajflt', int),
        ('utime', int), ('stime', int), ('cutime', int), ('cstime', int),
        ('priority', int), ('nice', int), ('num_threads', int),
        ('itrealvalue', int), ('starttime', int),
        ('vsize', int), ('rss', int), ('rsslim', int), 
        ('startcode', int), ('endcode', int), ('startstack', int),
        ('kstkesp', int), ('kstkeip', int),
        # obselete, use /proc/[pid]/status.
        ('signal', int), ('blocked', int), ('sigignore', int),
        ('sigcatch', int),
        # process waiting channel.
        ('wchan', int),
        # not maintained.
        ('nswap', int), ('cnswap', int),
        # signal to parent process when die.
        ('exit_signal', int),
        ('processor', int), ('rt_priority', int),
        ('policy', int), ('delayacct_blkio_ticks', int),
        ('guest_time', int), ('cguest_time', int),
    ]

    SETTING_KEYS = ['ibcthread']
    ENVAR_KEYS = ['KMP_AFFINITY']

    CPU_NAMES = ['us', 'sy', 'ni', 'id', 'wa', 'hi', 'si', 'st']

    @classmethod
    def get_pstat(cls):
        pid = os.getpid()
        f = open('/proc/%d/stat'%pid)
        sinfo = f.read().split()
        f.close()
        pstat = dict()
        for it in range(len(cls.PSTAT_KEYS)):
            key, typ = cls.PSTAT_KEYS[it]
            if it >= len(sinfo):
                value = None
            else:
                value = typ(sinfo[it])
            pstat[key] = value
        return pstat

    @staticmethod
    def get_cpu_frame():
        f = open('/proc/stat')
        totcpu = f.readlines()[0]
        f.close()
        return [float(it) for it in totcpu.split()[1:]]

    @staticmethod
    def calc_cpu_difference(frame0, frame1):
        frame = list()
        for it in range(len(frame0)):
            frame.append(frame1[it]-frame0[it])
        return frame

    @classmethod
    def get_envar(cls):
        envar = dict()
        for key in cls.ENVAR_KEYS:
            envar[key] = os.environ.get(key, None)
        return envar

    @staticmethod
    def get_loadavg():
        f = open('/proc/loadavg')
        loadavg = f.read()
        f.close()
        return [float(val) for val in loadavg.split()[:3]]

    def __init__(self, svr, **kw):
        self.cputotal = kw.pop('cputotal', True)
        self.jiffytime = kw.pop('jiffytime', 0.01)
        self.records = list()
        super(RuntimeStatAnchor, self).__init__(svr, **kw)

    def _get_record(self):
        record = dict()
        record['time'] = time.time()
        pstat = self.get_pstat()
        cpu = self.get_cpu_frame()
        loadavg = self.get_loadavg()
        # pstat.
        for key in ['utime', 'stime', 'priority', 'nice', 'num_threads',
                'vsize', 'rss', 'rt_priority']:
            record[key] = pstat[key]
        # cpu usage.
        record['cpu'] = cpu
        # loadavg.
        record['loadavg'] = loadavg
        # envar.
        record.update(self.get_envar())
        # timer.
        record.update(self.svr.timer)
        return record

    def _msg_setting(self, record):
        return ' '.join([
            '%s=%s' % (key, str(getattr(self.svr, key, None))) for key in
                self.SETTING_KEYS
        ])

    def _msg_envar(self, record):
        return ' '.join([
            '%s=%s' % (key, str(record[key])) for key in self.ENVAR_KEYS
        ])

    def _msg_cpu(self, record):
        # get the difference to the frame since last run of this method.
        if len(self.records):
            oldtime = self.records[-1]['time']
            framediff = self.calc_cpu_difference(
                self.records[-1]['cpu'], record['cpu'])
            pcpudiff = [
                record['utime'] - self.records[-1]['utime'],
                record['stime'] - self.records[-1]['stime'],
            ]
        else:
            oldtime = record['time']
            framediff = record['cpu']
            pcpudiff = [record['utime'], record['stime']]
        # calculate the percentage.
        if self.cputotal:
            jiffy = sum(framediff)
        else:
            jiffy = (record['time']-oldtime)/self.jiffytime
        if jiffy == 0.0: jiffy = 1.e100
        pscale = [it/jiffy*100 for it in pcpudiff]
        oscale = [it/jiffy*100 for it in framediff]
        # build message.
        process = '%.2f%%utime %.2f%%stime' % (pscale[0], pscale[1])
        overall = ' '.join(['%s%s' % ('%.2f%%'%oscale[it], self.CPU_NAMES[it])
            for it in range(len(self.CPU_NAMES))
        ])
        return ' '.join(['cputotal=%s'%self.cputotal, process, overall])
    @staticmethod
    def _parse_cpu(line):
        every = line.split()
        time = float(every[0])
        return [time] + [float(tok.split('%')[0]) for tok in every[2:]]
    @classmethod
    def plot_cpu(cls, lines, ax, xtime=False, showx=True,
            lloc='right'):
        arr, xval, xlabel = cls._parse(lines, 'cpu', xtime)
        arr[0,:] = arr[1,:]
        ax.plot(xval, arr[:,2:5].sum(axis=1), '-', label='us+st+ni')
        ax.plot(xval, arr[:,5:7].sum(axis=1), ':', label='id+wa')
        ax.plot(xval, arr[:,0:2].sum(axis=1), '--', label='utime+stime')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('CPU %')
        ax.set_ylim([0,100])
        ax.legend(loc=lloc)

    def _msg_mem(self, record):
        return '%d' % record['vsize']
    @staticmethod
    def _parse_mem(line):
        time, vsize = line.split()
        return [float(time), int(vsize)]
    @classmethod
    def plot_mem(cls, lines, ax, xtime=False, showx=True,
            lloc=None):
        arr, xval, xlabel = cls._parse(lines, 'mem', xtime)
        ax.plot(xval, arr[:,0]/1024**2, '-')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('Memory usage (MB)')

    def _msg_loadavg(self, record):
        return '%.2f %.2f %.2f' % tuple(record['loadavg'])
    @staticmethod
    def _parse_loadavg(line):
        return [float(val) for val in line.split()]
    @classmethod
    def plot_loadavg(cls, lines, ax, xtime=False, showx=True,
            lloc='right'):
        arr, xval, xlabel = cls._parse(lines, 'loadavg', xtime)
        ax.plot(xval, arr[:,0], '-', label='1 min')
        ax.plot(xval, arr[:,1], ':', label='5 min')
        ax.plot(xval, arr[:,2], '--', label='15 min')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('Load average')
        ax.legend(loc=lloc)

    @classmethod
    def _parse(cls, lines, key, xtime):
        myhead = 'RT_%s: ' % key
        nmyhead = len(myhead)
        mymethod = getattr(cls, '_parse_%s' % key)
        data = list()
        for line in lines:
            loc = line.find(myhead)
            if loc > -1:
                data.append(mymethod(line[loc+nmyhead:]))
        arr = np.array(data, dtype='float64')
        xval = arr[:,0]-arr[0,0] if xtime else np.arange(arr.shape[0])+1
        xlabel = 'Time (s)' if xtime else 'Steps'
        return arr[:,1:].copy(), xval.copy(), xlabel

    def postfull(self):
        if not sys.platform.startswith('linux'): return
        rec = self._get_record()
        # output the messages.
        time = rec['time']
        for mkey in ['cpu', 'loadavg', 'mem', 'setting', 'envar']:
            method = getattr(self, '_msg_%s'%mkey)
            self.svr.mesg('RT_%s: %.20e %s\n' % (mkey, time, method(rec)))
        # save.
        self.records.append(rec)

class MarchStatAnchor(Anchor):
    """
    Report the time used in each methods of marching.
    """

    @classmethod
    def plot(cls, keys, lines, ax, lloc='best'):
        maxavg = 0.0
        for key in keys:
            arr, xval, xlabel = cls.parse(lines, key)
            ax.plot(xval, arr[:,0], label='%s'%key)
            maxavg = max(arr[:,0].mean(), maxavg)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Time (s)')
        ax.set_ylim([0.0, maxavg*1.5])
        ax.legend(loc=lloc)

    @classmethod
    def parse(cls, lines, key, diff=True):
        myhead = 'MA_%s: ' % key
        nmyhead = len(myhead)
        data = list()
        for line in lines:
            loc = line.find(myhead)
            if loc > -1:
                data.append([float(val) for val in line[loc+nmyhead:].split()])
        arr = np.array(data, dtype='float64')
        if diff:
            arr[1:,:] = arr[1:,:] - arr[:-1,:]
            arr[0,:] = arr[1,:]
        xval = np.arange(arr.shape[0])+1
        xlabel = 'Steps'
        return arr, xval.copy(), xlabel

    def postfull(self):
        for key in ['march'] + self.svr.mmnames:
            val = self.svr.timer.get(key, None)
            if val == None:
                continue
            self.svr.mesg('MA_%s: %g\n' % (key, val))

class TpoolStatAnchor(Anchor):
    """
    Report the ticks used in each threads in pool.
    """

    @classmethod
    def plot(cls, key, lines, ax, showx=True, lloc='best'):
        arr, xval, xlabel = cls._parse(lines, key)
        for it in range(arr.shape[1]):
            ax.plot(xval, arr[:,it], label='%s%d'%(key, it))
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('CPU ticks')
        ax.legend(loc=lloc)

    @classmethod
    def _parse(cls, lines, key):
        myhead = 'TP_%s: ' % key
        nmyhead = len(myhead)
        data = list()
        for line in lines:
            loc = line.find(myhead)
            if loc > -1:
                data.append([int(val) for val in line[loc+nmyhead:].split()])
        arr = np.array(data, dtype='int32')
        arr[1:,:] = arr[1:,:] - arr[:-1,:]
        arr[0,:] = arr[1,:]
        xval = np.arange(arr.shape[0])+1
        xlabel = 'Steps'
        return arr, xval.copy(), xlabel

    def postfull(self):
        for key in self.svr.mmnames:
            if key not in getattr(self.svr, 'ticker', []):
                continue
            vals = self.svr.ticker[key]
            nval = len(vals)
            self.svr.mesg('TP_%s: %s\n' % (
                key, ' '.join(['%d'%val for val in vals])
            ))

################################################################################
# Initialization.
################################################################################

class FillAnchor(Anchor):
    """
    Fill the array with value.
    """
    def __init__(self, svr, **kw):
        self.keys = kw.pop('keys')
        self.value = kw.pop('value')
        super(FillAnchor, self).__init__(svr, **kw)
    def provide(self):
        for key in self.keys:
            getattr(self.svr, key).fill(self.value)

################################################################################
# Glue.
################################################################################

class GlueAnchor(Anchor):
    """
    Use Glue class to glue specified BC objects of a solver object.

    @cvar KEYS_ENABLER: names of the arrays that should be modified when
        enabling/disabling the glue.
    @ctype KEYS_ENABLER: sequence
    @ivar bcpairs: a sequence of 2-tuples for BC object pairs to be glued.
    @itype bcpairs: sequence
    """

    KEYS_ENABLER = tuple()

    def __init__(self, svr, **kw):
        self.bcpairs = kw.pop('bcpairs')
        super(GlueAnchor, self).__init__(svr, **kw)

    def _attach_glue(self):
        """
        Attach Glue objects to specified BC object pairs.

        @return: nothing
        """
        nmbc = dict([(bc.name, bc) for bc in self.svr.bclist])
        for key0, key1 in self.bcpairs:
            assert nmbc[key0].glue is None
            assert nmbc[key1].glue is None
            boundcond.Glue(nmbc[key0], nmbc[key1])

    def _detach_glue(self):
        """
        Detach Glue objects from specified BC object pairs.

        @return: nothing
        """
        nmbc = dict([(bc.name, bc) for bc in self.svr.bclist])
        for key0, key1 in self.bcpairs:
            assert isinstance(nmbc[key0].glue, boundcond.Glue)
            assert isinstance(nmbc[key1].glue, boundcond.Glue)
            nmbc[key0].glue = None
            nmbc[key1].glue = None

    def _enable_glue(self, check=True):
        """
        Enable the gluing mechanism by calling Glue.enable() for specified BC
        object pairs.

        @keyword check: check Glue object or not.  Default True.
        @type check: bool
        @return: nothing
        """
        svr = self.svr
        if check:
            self._attach_glue()
        nmbc = dict([(bc.name, bc) for bc in svr.bclist])
        for keys in self.bcpairs:
            for key in keys:
                nmbc[key].glue.enable(*self.KEYS_ENABLER)
        svr._clib_cuse_c.prepare_ce(ctypes.byref(svr.exd))
        svr._clib_cuse_c.prepare_sf(ctypes.byref(svr.exd))
        if svr.scu: svr.cumgr.arr_to_gpu()

    def _disable_glue(self, check=True):
        """
        Disable the gluing mechanism by calling Glue.disable() for specified BC
        object pairs.

        @keyword check: check Glue object or not.  Default True.
        @type check: bool
        @return: nothing
        """
        svr = self.svr
        nmbc = dict([(bc.name, bc) for bc in svr.bclist])
        for keys in self.bcpairs:
            for key in keys:
                nmbc[key].glue.disable(*self.KEYS_ENABLER)
        svr._clib_cuse_c.prepare_ce(ctypes.byref(svr.exd))
        svr._clib_cuse_c.prepare_sf(ctypes.byref(svr.exd))
        if svr.scu: svr.cumgr.arr_to_gpu()
        if check:
            self._detach_glue()
