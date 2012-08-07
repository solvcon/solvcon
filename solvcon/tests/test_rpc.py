# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2012 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.
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

from unittest import TestCase

class Solver(object):
    def __init__(self, msg, msg_other):
        self.msg = msg
        self.msg_other = msg_other

    def __str__(self):
        return self.msg

    def task(self):
        pass

    def assert_msg(self, msg):
        assert self.msg == msg

    def send_msg(self, worker=None):
        serial = worker.serial
        for peern in worker.pconns.keys():
            if isinstance(peern, int) and peern != serial:
                break
        pconn = worker.pconns[peern]
        pconn.send(self.msg)

    def recv_msg(self, worker=None):
        serial = worker.serial
        for peern in worker.pconns.keys():
            if isinstance(peern, int) and peern != serial:
                break
        pconn = worker.pconns[peern]
        msg = pconn.recv()
        assert msg == self.msg_other

class TestWorker(TestCase):
    def test_hire(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..rpc import Worker, Dealer
        dealer = Dealer()
        for iproc in range(2):
            dealer.hire(Worker(None))
        dealer.terminate()

    def test_set_muscle(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..rpc import Worker, Dealer
        dealer = Dealer()
        for iproc in range(2):
            dealer.hire(Worker(None))
        muscles = [Solver("solver0", "solver1"), Solver("solver1", "solver0")]
        for iproc in range(2):
            sdw = dealer[iproc]
            sdw.remote_setattr('muscle', muscles[iproc])
        dealer[0].cmd.assert_msg('solver0')
        dealer[1].cmd.assert_msg('solver1')
        dealer.terminate()

    def test_cmd(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..rpc import Worker, Dealer
        dealer = Dealer()
        muscles = [Solver("solver0", "solver1"), Solver("solver1", "solver0")]
        for iproc in range(2):
            dealer.hire(Worker(muscles[iproc]))
        for iproc in range(2):
            dealer[iproc].cmd.task()
        dealer.terminate()

    def test_bridge_and_sendrecv(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..rpc import Worker, Dealer
        dealer = Dealer()
        muscles = [Solver("solver0", "solver1"), Solver("solver1", "solver0")]
        for iproc in range(2):
            dealer.hire(Worker(muscles[iproc]))
        dealer.bridge((0,1))
        dealer.barrier()
        dealer[0].cmd.send_msg(with_worker=True)
        dealer[1].cmd.recv_msg(with_worker=True)
        dealer.barrier()
        dealer.terminate()

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
