# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Anchors attached to the solvers.
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
    def prefull(self):
        pass
    def prehalf(self):
        pass
    def premarchsol(self):
        pass
    def preexsoln(self):
        pass
    def prebcsoln(self):
        pass
    def precfl(self):
        pass
    def premarchdsol(self):
        pass
    def preexdsoln(self):
        pass
    def prebcdsoln(self):
        pass
    def posthalf(self):
        pass
    def postfull(self):
        pass
    def postloop(self):
        pass
    def exhaust(self):
        pass

class AnchorList(list):
    """
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
            getattr(anchor, method)()

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
    def plot_cpu(cls, lines, ax, xtime=False, showx=True):
        arr, xval, xlabel = cls._parse(lines, 'cpu', xtime)
        ax.plot(xval, arr[:,2:5].sum(axis=1), '-', label='us+st+ni')
        ax.plot(xval, arr[:,5:7].sum(axis=1), ':', label='id+wa')
        ax.plot(xval, arr[:,0:2].sum(axis=1), '--', label='utime+stime')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('CPU %')
        ax.legend(loc='right')

    def _msg_march(self, record):
        return '%g %g %g %g %g %g %g %g' % (
            record['march'], record['update'],
            record['msol'], record['ibcsol'], record['bcsol'],
            record['mdsol'], record['ibcdsol'], record['bcdsol'],
        )
    @staticmethod
    def _parse_march(line):
        return [float(tok) for tok in line.split()]
    @classmethod
    def plot_march(cls, lines, ax, xtime=False, showx=True):
        arr, xval, xlabel = cls._parse(lines, 'march', xtime)
        arr[1:,:] = arr[1:,:] - arr[:-1,:]
        ax.plot(xval, arr[:,0], '-', label='march')
        ax.plot(xval, arr[:,2], '--', label='msol')
        ax.plot(xval, arr[:,5], ':', label='mdsol')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('March (s)')
        ax.legend(loc='upper left')
    @classmethod
    def plot_marchother(cls, lines, ax, xtime=False, showx=True):
        arr, xval, xlabel = cls._parse(lines, 'march', xtime)
        arr[1:,:] = arr[1:,:] - arr[:-1,:]
        ax.plot(xval, arr[:,1], '-', label='update')
        ax.plot(xval, arr[:,3], '+', label='ibcsol')
        ax.plot(xval, arr[:,6], 'x', label='ibcdsol')
        ax.plot(xval, arr[:,4], '--', label='bcsol')
        ax.plot(xval, arr[:,7], ':', label='bcdsol')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('March other (s)')
        ax.legend(loc='upper left')
    @classmethod
    def plot_perf(cls, lines, ax, xtime=False, showx=True):
        arr, xval, xlabel = cls._parse(lines, 'march', xtime)
        arr[1:,:] = arr[1:,:] - arr[:-1,:]
        time = arr[:,0]
        ax.plot(xval, 1./time, '-')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('Performance (iter/s)')

    def _msg_mem(self, record):
        return '%d' % record['vsize']
    @staticmethod
    def _parse_mem(line):
        time, vsize = line.split()
        return [float(time), int(vsize)]
    @classmethod
    def plot_mem(cls, lines, ax, xtime=False, showx=True):
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
    def plot_loadavg(cls, lines, ax, xtime=False, showx=True):
        arr, xval, xlabel = cls._parse(lines, 'loadavg', xtime)
        ax.plot(xval, arr[:,0], '-', label='1 min')
        ax.plot(xval, arr[:,1], ':', label='5 min')
        ax.plot(xval, arr[:,2], '--', label='15 min')
        if showx: ax.set_xlabel(xlabel)
        ax.set_ylabel('Load average')
        ax.legend(loc='right')

    @classmethod
    def _parse(cls, lines, key, xtime):
        from numpy import array, arange
        myhead = 'RT_%s' % key
        mymethod = getattr(cls, '_parse_%s' % key)
        arr = array([mymethod(line.split(' ', 1)[1]) for line in
            lines if line.startswith(myhead)
        ], dtype='float64')
        xval = arr[:,0]-arr[0,0] if xtime else arange(arr.shape[0])+1
        xlabel = 'Time (s)' if xtime else 'Steps'
        return arr[:,1:].copy(), xval.copy(), xlabel

    def postfull(self):
        import sys
        if not sys.platform.startswith('linux'): return
        rec = self._get_record()
        # output the messages.
        time = rec['time']
        for mkey in ['march', 'cpu', 'loadavg', 'mem', 'envar']:
            method = getattr(self, '_msg_%s'%mkey)
            self.svr.mesg('RT_%s: %.20e %s\n' % (mkey, time, method(rec)))
        # save.
        self.records.append(rec)

class ZeroIAnchor(Anchor):
    """
    Fill the solutions with zero.
    """
    def provide(self):
        self.svr.soln.fill(0.0)
        self.svr.dsoln.fill(0.0)
