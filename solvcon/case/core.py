# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Core components for simulation cases.
"""

from solvcon.gendata import SingleAssignDict, AttributeDict

class ArrangementRegistry(SingleAssignDict, AttributeDict):
    """
    Arrangement registry class.  A simulation is a callable that returns a case
    object.
    """
    def __setitem__(self, key, value):
        assert callable(value)
        super(ArrangementRegistry, self).__setitem__(key, value)
arrangements = ArrangementRegistry()  # overall registry singleton.

class CaseInfoMeta(type):
    """
    Meta class for case information class.
    """
    def __new__(cls, name, bases, namespace):
        newcls = super(CaseInfoMeta, cls).__new__(cls, name, bases, namespace)
        # incremental modification of defdict.
        defdict = {}
        for base in bases:
            defdict.update(getattr(base, 'defdict', {}))
        defdict.update(newcls.defdict)
        newcls.defdict = defdict
        # create different simulation registry objects for case classes.
        newcls.arrangements = ArrangementRegistry()
        return newcls
class CaseInfo(dict):
    """
    Generic case information abstract class.  It's the base class that all case
    information classes should subclass, to form hierarchical information 
    object.
    """
    __metaclass__ = CaseInfoMeta
    defdict = {}
    def __getattr__(self, name):
        """
        Consult self dictionary for attribute.  It's a shorthand.
        """
        return self[name]
    def __setattr__(self, name, value):
        """
        Save to self dictionary first, then self object table.

        @note: This method is overriden as a stupid-preventer.  It makes
        attribute setting consistent with attribute getting.
        """
        if name in self:
            self[name] = value
        else:
            super(CaseInfo, self).__setattr__(name, value)
    def _set_through(self, key, val):
        """
        Set to self with the dot-separated key.
        """
        tokens = key.split('.', 1)
        fkey = tokens[0]
        if len(tokens) == 2:
            self[fkey]._set_through(tokens[1], val)
        else:
            self[fkey] = val
    def __init__(self, _defdict=None, *args, **kw):
        """
        Assign default values to self after initiated.

        @keyword _defdict: customized defdict; internal use only.
        @type _defdict: dict
        """
        super(CaseInfo, self).__init__(*args, **kw)
        # customize defdict.
        if _defdict is None:
            defdict = self.defdict
        else:
            defdict = dict(self.defdict)
            defdict.update(_defdict)
        # parse first hierarchy to form key groups.
        keygrp = dict()
        for key in defdict.keys():
            if key is None or key == '':
                continue
            tokens = key.split('.', 1)
            if len(tokens) == 2:
                fkey, rkey = tokens
                keygrp.setdefault(fkey, dict())[rkey] = defdict[key]
            else:
                fkey = tokens[0]
                keygrp[fkey] = defdict[fkey]
        # set up first layer keys recursively.
        for fkey in keygrp.keys():
            data = keygrp[fkey]
            if isinstance(data, dict):
                self[fkey] = CaseInfo(_defdict=data)
            elif isinstance(data, type):
                try:
                    self[fkey] = data()
                except TypeError:
                    self[fkey] = data
            else:
                self[fkey] = data

class BaseCase(CaseInfo):
    """
    Base class for simulation cases.

    @ivar _have_init: flag that self was initialized or not.
    @itype _have_init: bool
    """

    from .. import conf, batch
    defdict = {
        # execution related.
        'execution.fpdtype': conf.env.fpdtypestr,
        'execution.runhooks': list,
        'execution.run_inner': False,
        'execution.scheduler': batch.Scheduler,
        'execution.resources': dict,    # for scheduler.
        'execution.time': 0.0,
        'execution.time_increment': 0.0,
        'execution.step_init': 0,
        'execution.step_current': None,
        'execution.steps_run': None,
        'execution.steps_stride': 1,
        'execution.cCFL': 0.0,  # current.
        'execution.aCFL': 0.0,  # accumulated.
        'execution.mCFL': 0.0,  # mean.
        'execution.neq': 0, # number of unknowns.
        'execution.var': dict,  # for Calculator hooks.
        'execution.varstep': None,  # the step for which var and dvar are valid.
        # io related.
        'io.abspath': False,    # flag to use abspath or not.
        'io.rootdir': None,
        'io.basedir': None,
        'io.basefn': None,
        'io.empty_jobdir': False,
        # conditions.
        'condition.mtrllist': list,
        # solver.
        'solver.solvertype': None,
        'solver.solverobj': None,
        # logging.
        'log.time': dict,
    }
    del conf, batch
    from solvcon.helper import info
    info = staticmethod(info)

    def _log_start(self, action, msg='', postmsg=' ... '):
        """
        Print to user and record start time for certain action.

        @param action: action key.
        @type action: str
        @keyword msg: trailing message for the action key.
        @type msg: str
        @return: nothing.
        """
        from time import time
        tarr = [0,0,0]
        tarr[0] = time()
        self.log.time[action] = tarr
        self.info('Start %s%s%s' % (action, msg, postmsg))

    def _log_end(self, action, msg=''):
        """
        Print to user and record end time for certain action.

        @param action: action key.
        @type action: str
        @keyword msg: supplemental message.
        @type msg: str
        @return: nothing
        """
        from time import time
        tarr = self.log.time.setdefault(action, [0,0,0])
        tarr[1] = time()
        tarr[2] = tarr[1] - tarr[0]
        self.info('done.%s time (sec) = %g\n' % (msg, tarr[2]))

    def __init__(self, **kw):
        """
        Initiailize the basic case.  Set through keyword parameters.
        """
        import os
        super(BaseCase, self).__init__(**kw)
        # populate value from keywords.
        for cinfok in self.defdict.keys():
            lkey = cinfok.split('.')[-1]
            val = kw.get(lkey, None)
            if val is not None:
                self._set_through(cinfok, val)
        # expand basedir.
        if self.io.abspath:
            self.io.basedir = os.path.abspath(self.io.basedir)
        # second-phase initilization flag.
        self._have_init = False

    def init(self, force=False):
        """
        Second-phase initialization.  This initilization should be performed 
        right before the running of the case.  Subclass should set _have_init
        attribute on the end of this method.

        @keyword force: flag to force initialization no matter self was
            initialized or not.
        @type force: bool
        @return: flag initialized or not.
        @rtype: bool
        """
        assert not self._have_init or force
        self._have_init = True
        return self._have_init

    def _runhooks(self, method):
        """
        Invoke the specified method for each runhook.
        
        Note: the order of execution of final hooks is reversed.

        @param method: name of the method to run.
        @type method: str
        """
        runhooks = self.execution.runhooks
        if method == 'postloop':
            runhooks = reversed(runhooks)
        for hook in self.execution.runhooks:
            getattr(hook, method)()

    def run(self):
        """
        Run the simulation case; time marching.  Subclass should implement
        this.

        @return: nothing.
        """
        raise NotImplementedError

    @classmethod
    def register_arrangement(cls, func):
        """
        Decorate simulation functions.  This function assert required signature
        which is necessary for a function to be a valid simulation function.
        Moreover, all the simulation function should be decorated by this
        decorator.

        @return: simulation function.
        @rtype: callable
        """
        from ..batch import Scheduler
        def simu(*args, **kw):
            kw.pop('casename', None)
            submit = kw.pop('submit', None)
            resources = kw.pop('resources', dict())
            scheduler = kw.get('scheduler', Scheduler)
            casename = func.__name__
            case = func(casename=casename, *args, **kw)
            if submit:
                sbm = scheduler(case, arnname=casename, **resources)
                sbm()
            else:
                case.init()
                case.run()
        # register self to simulation registries.
        cls.arrangements[func.__name__] = simu
        arrangements[func.__name__] = simu
        return simu

class Hook(object):
    """
    Container class for various hooking subroutines for BaseCase.

    @ivar case: case object.
    @itype case: BaseCase
    @ivar info: information output function.
    @itype info: callable
    @ivar psteps: the interval number of steps between printing.
    @itype psteps: int
    @ivar kws: excessive keywords.
    @itype kws: dict
    """
    def __init__(self, case, **kw):
        """
        @param case: case object.
        @type case: BaseCase
        """
        assert isinstance(case, BaseCase)
        self.case = case
        self.info = case.info
        self.psteps = kw.pop('psteps', None)
        # save excessive keywords.
        self.kws = dict(kw)
        super(Hook, self).__init__()

    def _makedir(self, dirname, verbose=False):
        """
        Make new directory if it does not exist in prior.

        @param dirname: name of directory to be created.
        @type dirname: str
        @keyword verbose: flag if print out creation message.
        @type verbose: bool
        """
        import os
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            if verbose:
                self.info('Created %s' % dirname)

    def _depend(self, deplst, verbose=False, stop_on_false=True):
        """
        Check for dependency to another hook.

        @param deplst: list of depended hook classes.
        @type deplst: list
        @keyword verbose: flag print message.
        @type verbose: bool
        @keyword stop_on_false: flag stop on false.
        @type stop_on_false: bool
        @return: dependency met or not.
        @rtype: bool
        """
        hooks = self.case.execution.runhooks
        info = self.info
        # check.
        metlst = []
        msglst = []
        for ahook in deplst:
            metlst.append(False)
            for obj in hooks:
                if isinstance(obj, ahook):
                    metlst[-1] = True
                    break
            if not metlst[-1]:
                msglst.append("%s should be enabled for %s." % (
                    ahook.__name__, self.__class__.__name__))
        if verbose and msglst:
            info('\n'.join(msglst)+'\n')
        if stop_on_false and msglst:
            raise RuntimeError, '\n'.join(msglst)
        # return.
        for met in metlst:
            if not met:
                return False
        return True

    def preloop(self):
        """
        Things to do before the time-marching loop.
        """
        pass

    def premarch(self):
        """
        Things to do before the time march for a specific time step.
        """
        pass

    def postmarch(self):
        """
        Things to do after the time march for a specific time step.
        """
        pass

    def postloop(self):
        """
        Things to do after the time-marching loop.
        """
        pass

class ProgressHook(Hook):
    """
    Print progess.

    @ivar linewidth: the maximal width for progress symbol.  50 is upper limit.
    @itype linewidth: int
    """

    def __init__(self, case, **kw):
        self.linewidth = kw.pop('linewidth', 50)
        assert self.linewidth <= 50
        super(ProgressHook, self).__init__(case, **kw)

    def preloop(self):
        istep = self.case.execution.step_current
        nsteps = self.case.execution.steps_run
        info = self.info
        info("Steps to run: %d, current step: %d\n" % (nsteps, istep))

    def postmarch(self):
        from time import time
        istep = self.case.execution.step_current
        nsteps = self.case.execution.steps_run
        tstart = self.case.log.time['loop_march'][0]
        psteps = self.psteps
        linewidth = self.linewidth
        cCFL = self.case.execution.cCFL
        info = self.info
        # calculate estimated remaining time.
        tcurr = time()
        tleft = (tcurr-tstart) * ((float(nsteps)-float(istep))/float(istep))
        # output information.
        if istep%psteps == 0:
            info("#")
        if istep > 0 and istep%(psteps*linewidth) == 0:
            info(" %d/%d (%.1fs left) %.2f\n" % (istep, nsteps, tleft, cCFL))
        elif istep == nsteps:
            info(" %d/%d done\n" % (istep, nsteps))

class CflHook(Hook):
    """
    Make sure is CFL number is bounded and print averaged CFL number over time.

    @ivar cflmin: CFL number should be greater than or equal to the value.
    @itype cflmin: float
    @ivar cflmax: CFL number should be less than the value.
    @itype cflmax: float
    @ivar fullstop: flag to stop when CFL is out of bound.  Default True.
    @itype fullstop: bool
    """

    def __init__(self, case, **kw):
        self.cflmin = kw.pop('cflmin', 0.0)
        self.cflmax = kw.pop('cflmax', 1.0)
        self.fullstop = kw.pop('fullstop', True)
        super(CflHook, self).__init__(case, **kw)

    def _notify(self, msg):
        from warnings import warn
        if self.fullstop:
            raise RuntimeError, msg
        else:
            warn(msg)

    def postmarch(self):
        cCFL = self.case.execution.cCFL
        istep = self.case.execution.step_current
        if self.cflmin != None and cCFL < self.cflmin:
            self._notify("CFL = %g < %g after step: %d" % (
                cCFL, self.cflmin, istep))
        if self.cflmax != None and cCFL >= self.cflmax:
            self._notify("CFL = %g >= %g after step: %d" % (
                cCFL, self.cflmax, istep))

    def postloop(self):
        info = self.info
        info("Averaged maximum CFL = %g.\n" % self.case.execution.mCFL)
