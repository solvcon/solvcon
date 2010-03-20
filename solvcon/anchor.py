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
    def __init__(self, svr, **kw):
        self.reports = kw.pop('reports',
            ['time', 'mem', 'loadavg', 'cpu', 'envar'])
        self.cputotal = kw.pop('cputotal', True)
        self.cputime = 0.0
        self.cpuframe = None
        self.jiffytime = 0.01
        super(RuntimeStatAnchor, self).__init__(svr, **kw)

    def _RT_envar(self):
        import os
        msgs = list()
        for key in ['KMP_AFFINITY']:
            msgs.append('%s=%s' % (key, str(os.environ.get(key, None))))
        return 'envar: %s' % ' '.join(msgs)

    def _RT_time(self):
        from time import time
        return 'time: %.20e' % time()

    def _RT_mem(self):
        import os
        # read information.
        f = open('/proc/%d/status' % os.getpid())
        status = f.read().strip()
        f.close()
        # format.
        status = dict([line.split(':') for line in status.split('\n')])
        newstatus = dict()
        for want in ['VmPeak', 'VmSize', 'VmRSS']:
            newstatus[want] = int(status[want].strip().split()[0])
        status = newstatus
        # return.
        return 'mem: %d%s %d%s %d%s' % (
            status['VmPeak'], 'VmPeak',
            status['VmSize'], 'VmSize',
            status['VmRSS'], 'VmRSS',
        )

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

    def _RT_cpu(self):
        import time
        names = ['us', 'sy', 'ni', 'id', 'wa', 'hi', 'si', 'st']
        # get the difference to the frame since last run of this method.
        currtime = time.time()
        if self.cpuframe:
            currframe = self.get_cpu_frame()
            framediff = self.calc_cpu_difference(self.cpuframe, currframe)
        else:
            framediff = currframe = self.get_cpu_frame()
        # calculate the percentage.
        if self.cputotal:
            jiffy = sum(framediff)
        else:
            jiffy = (currtime-self.cputime)/self.jiffytime
        if jiffy == 0.0: jiffy = 1.e100
        scale = [it/jiffy*100 for it in framediff]
        # build message.
        msgs = list()
        for it in range(len(names)):
            msgs.append('%s%s' % ('%.2f%%'%scale[it], names[it]))
        # housekeeping.
        self.cputime = currtime
        self.cpuframe = currframe
        return 'cpu: ' + ' '.join(msgs)

    def _RT_loadavg(self):
        f = open('/proc/loadavg')
        loadavg = f.read()
        f.close()
        return 'loadavg: %s' % ' '.join(loadavg.split()[:3])

    def postfull(self):
        import sys
        if not sys.platform.startswith('linux'): return
        for rep in self.reports:
            self.svr.mesg('RT_'+getattr(self, '_RT_'+rep)()+'\n')

class ZeroIAnchor(Anchor):
    """
    Fill the solutions with zero.
    """
    def provide(self):
        self.svr.soln.fill(0.0)
        self.svr.dsoln.fill(0.0)
