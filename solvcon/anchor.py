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
Anchors attached to the solvers.  There's only one base anchor class for
subclassing.  Any other anchors defined here are for directly installation.
"""

class Anchor(object):
    """
    Anchor that called by solver objects at various stages.

    @ivar svr: the solver object to be attached to.
    @itype svr: solvcon.solver.Solver
    @ivar kws: excessive keywords.
    @itype kws: dict
    """

    def __init__(self, svr, **kw):
        from .solver import BaseSolver
        assert isinstance(svr, BaseSolver)
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
        from .io.vtkxml import VtkXmlUstGridWriter
        from .solver import FakeBlockVtk
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
        wtr = VtkXmlUstGridWriter(FakeBlockVtk(self.svr), fpdtype=self.fpdtype,
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

################################################################################
# Anchors for in situ visualization.
################################################################################

class VtkAnchor(Anchor):
    """
    Abstract class for VTK filtering anchor.  Must override process() method
    for use.

    @ivar anames: the arrays in der of solvers to be saved.  True means in der.
    @itype anames: dict
    @ivar fpdtype: string for floating point data type (in numpy convention).
    @itype fpdtype: str
    @ivar psteps: the interval (in step) to save data.
    @itype psteps: int
    @ivar vtkfn_tmpl: the template string for the VTK file.
    @itype vtkfn_tmpl: str
    """
    VANMAP = dict(float32='vtkFloatArray', float64='vtkDoubleArray')
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
    ############################################################################
    # Utilities methods.
    ############################################################################
    @staticmethod
    def _valid_vector(arr):
        """
        A valid vector must have 3 compoments.  If it has only 2, pad it.  If
        it has more than 3, raise ValueError.

        @param arr: input vector array.
        @type arr: numpy.ndarray
        @return: validated array.
        @rtype: numpy.ndarray
        """
        from numpy import empty
        if arr.shape[1] < 3:
            arrn = empty((arr.shape[0], 3), dtype=arr.dtype)
            arrn[:,2] = 0.0
            try:
                arrn[:,:2] = arr[:,:]
            except ValueError, e:
                args = e.args[:]
                args.append('arrn.shape=%s, arr.shape=%s' % (
                    str(arrn.shape), str(arr.shape)))
                e.args = args
                raise
            arr = arrn
        elif arr.shape[1] > 3:
            raise IndexError('arr.shape[1] = %d > 3'%arr.shape[1])
        return arr
    def _set_arr(self, arr, name):
        """
        Set the data of a ndarray to vtk array and return the set vtk array.
        If the array of the specified name existed, use the existing array.

        @param arr: input array.
        @type arr: numpy.ndarray
        @param name: array name.
        @type name: str
        @return: the set VTK array object.
        """
        import vtk
        ust = self.svr.ust
        if ust.GetCellData().HasArray(name):
            vaj = ust.GetCellData().GetArray(name)
        else:
            vaj = getattr(vtk, self.VANMAP[self.fpdtype])()
            # prepare for vector.
            if len(arr.shape) > 1:
                vaj.SetNumberOfComponents(3)
            # set number of tuples to allocate.
            vaj.SetNumberOfTuples(arr.shape[0])
            # cache.
            vaj.SetName(name)
            ust.GetCellData().AddArray(vaj)
        # set data.
        nt = arr.shape[0]
        it = 0
        if len(arr.shape) > 1:
            while it < nt:
                vaj.SetTuple3(it, *arr[it])
                it += 1
        else:
            while it < nt:
                vaj.SetValue(it, arr[it])
                it += 1
        return vaj
    def _aggregate(self):
        """
        Aggregate data from solver object to VTK unstructured mesh.

        @return: nothing
        """
        ngstcell = self.svr.ngstcell
        # collect derived.
        for key in self.anames:
            # get the array.
            if self.anames[key]:
                arr = self.svr.der[key][ngstcell:]
            else:
                arr = getattr(self.svr, key)[ngstcell:]
            # set array in unstructured mesh.
            if len(arr.shape) == 1:
                self._set_arr(arr, key)
            elif arr.shape[1] == self.svr.ndim:
                self._set_arr(self._valid_vector(arr), key)
            else:
                for it in range(arr.shape[1]):
                    self._set_arr(arr[:,it], '%s[%d]' % (key, it))
    ############################################################################
    # External interface.
    ############################################################################
    def preloop(self):
        self.process(0)
    def postmarch(self):
        psteps = self.psteps
        istep = self.svr.step_global
        if istep%psteps == 0:
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
        import os
        pid = os.getpid()
        f = open('/proc/%d/stat'%pid)
        sinfo = f.read().split()
        f.close()
        pstat = dict()
        for it in range(len(sinfo)):
            key, typ = cls.PSTAT_KEYS[it]
            pstat[key] = typ(sinfo[it])
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
        import os
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
        from time import time
        record = dict()
        record['time'] = time()
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
            '%s=%s' % (key, str(getattr(self.svr, key))) for key in
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
        from numpy import array, arange
        myhead = 'RT_%s: ' % key
        nmyhead = len(myhead)
        mymethod = getattr(cls, '_parse_%s' % key)
        data = list()
        for line in lines:
            loc = line.find(myhead)
            if loc > -1:
                data.append(mymethod(line[loc+nmyhead:]))
        arr = array(data, dtype='float64')
        xval = arr[:,0]-arr[0,0] if xtime else arange(arr.shape[0])+1
        xlabel = 'Time (s)' if xtime else 'Steps'
        return arr[:,1:].copy(), xval.copy(), xlabel

    def postfull(self):
        import sys
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
        from numpy import array, arange
        myhead = 'MA_%s: ' % key
        nmyhead = len(myhead)
        data = list()
        for line in lines:
            loc = line.find(myhead)
            if loc > -1:
                data.append([float(val) for val in line[loc+nmyhead:].split()])
        arr = array(data, dtype='float64')
        if diff:
            arr[1:,:] = arr[1:,:] - arr[:-1,:]
            arr[0,:] = arr[1,:]
        xval = arange(arr.shape[0])+1
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
        from numpy import array, arange
        myhead = 'TP_%s: ' % key
        nmyhead = len(myhead)
        data = list()
        for line in lines:
            loc = line.find(myhead)
            if loc > -1:
                data.append([int(val) for val in line[loc+nmyhead:].split()])
        arr = array(data, dtype='int32')
        arr[1:,:] = arr[1:,:] - arr[:-1,:]
        arr[0,:] = arr[1,:]
        xval = arange(arr.shape[0])+1
        xlabel = 'Steps'
        return arr, xval.copy(), xlabel

    def postfull(self):
        for key in self.svr.mmnames:
            if key not in self.svr.ticker:
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
