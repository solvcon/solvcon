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
Solvers that base on :py:class:`.mesh.Mesh`.
"""

from .case import CaseInfo

class MeshCase(CaseInfo):
    """
    :ivar runhooks: All the hook objects to be run.
    :type runhooks: solvcon.hook.HookList
    :ivar info: Message logger.
    :type info: solvcon.helper.Information

    Base class for simulation cases based on :py:class:`MeshSolver`.

    init() and run() are the two primary methods responsible for the
    execution of the simulation case object.  Both methods accept a keyword
    parameter ``level'' which indicates the run level of the run:

    - run level 0: fresh run (default),
    - run level 1: restart run,
    - run level 2: initialization only.
    """

    defdict = {
        # execution related.
        'execution.resources': dict,    # for batch.
        'execution.stop': False,
        'execution.time': 0.0,
        'execution.time_increment': 0.0,
        'execution.step_init': 0,
        'execution.step_current': None,
        'execution.steps_run': None,
        'execution.steps_stride': 1,
        'execution.marchret': None,
        'execution.neq': 0, # number of unknowns.
        'execution.var': dict,  # for Calculator hooks.
        'execution.varstep': None,  # the step for which var and dvar are valid.
        # io related.
        'io.abspath': False,    # flag to use abspath or not.
        'io.rootdir': None,
        'io.basedir': None,
        'io.basefn': None,
        'io.empty_jobdir': False,
        'io.solver_output': False,
        # conditions.
        'condition.mtrllist': list,
        # solver.
        'solver.solvertype': None,
        'solver.solverobj': None,
        # logging.
        'log.time': dict,
    }

    def __init__(self, **kw):
        """
        Initiailize the basic case.  Set through keyword parameters.
        """
        import os
        from .hook import HookList
        from .helper import Information
        # populate value from keywords.
        initpairs = list()
        for cinfok in self.defdict.keys():
            lkey = cinfok.split('.')[-1]
            initpairs.append((cinfok, kw.pop(lkey, None)))
        # initialize with the left keywords.
        super(MeshCase, self).__init__(**kw)
        # populate value from keywords.
        for cinfok, val in initpairs:
            if val is not None:
                self._set_through(cinfok, val)
        # create runhooks.
        self.runhooks = HookList(self)
        # expand basedir.
        if self.io.abspath:
            self.io.basedir = os.path.abspath(self.io.basedir)
        if self.io.basedir is not None and not os.path.exists(self.io.basedir):
            os.makedirs(self.io.basedir)
        # message logger.
        self.info = Information()

    def _log_start(self, action, msg='', postmsg=' ... '):
        """
        :param action: Action key.
        :type action: str
        :keyword msg: Trailing message for the action key.
        :type msg: str
        :return: Nothing.

        Print to user and record start time for certain action.
        """
        from time import time
        info = self.info
        tarr = [0,0,0]
        tarr[0] = time()
        self.log.time[action] = tarr
        # header.
        prefix = info.prefix * (info.width-info.level*info.nchar)
        info(prefix, travel=1)
        # content.
        info('\nStart %s%s%s' % (action, msg, postmsg))
        prefix = info.prefix * (info.width-info.level*info.nchar)
        info('\n' + prefix + '\n')

    def _log_end(self, action, msg='', postmsg=' . '):
        """
        :param action: Action key.
        :type action: str
        :keyword msg: Supplemental message.
        :type msg: str
        :return: Nothing

        Print to user and record end time for certain action.
        """
        from time import time
        info = self.info
        tarr = self.log.time.setdefault(action, [0,0,0])
        tarr[1] = time()
        tarr[2] = tarr[1] - tarr[0]
        # footer.
        prefix = info.prefix * (info.width-info.level*info.nchar)
        info(prefix + '\nEnd %s%s%sElapsed time (sec) = %g' % (
            action, msg, postmsg, tarr[2]))
        # up a level.
        prefix = info.prefix * (info.width-(info.level-1)*info.nchar)
        info('\n' + prefix + '\n', travel=-1)

    def init(self, level=0):
        """
        :keyword level: Run level; higher level does less work.
        :type level: int
        :return: Nothing

        An empty initializer for solvers.

        >>> cse = MeshCase()
        >>> cse.init()
        """
        pass

    def run(self, level=0):
        """
        :keyword level: Run level; higher level does less work.
        :type level: int
        :return: Nothing

        Temporal loop for the incorporated solver.

        >>> cse = MeshCase(basefn='meshcase')
        >>> cse.init()
        >>> cse.info.muted = True
        >>> cse.run()
        """
        # start log.
        self._log_start('run', msg=' '+self.io.basefn)
        self.info("\n")
        # prepare for time marching.
        self.execution.step_current = 0
        self.runhooks('preloop')
        self._log_start('loop_march')
        while self.execution.step_current < self.execution.steps_run:
            self.runhooks('premarch')
            self.execution.marchret = self.solver.solverobj.march(
                self.execution.step_current*self.execution.time_increment,
                self.execution.time_increment)
            self.execution.step_current += 1
            self.runhooks('postmarch')
        self._log_start('loop_march')
        self.runhooks('postloop')
        # end log.
        self._log_end('run')

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
