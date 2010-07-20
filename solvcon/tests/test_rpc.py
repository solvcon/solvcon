# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

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

class TestRemote(TestCase):
    def test_python(self):
        from subprocess import PIPE
        from ..rpc import Remote
        remote = Remote('localhost')
        self.assertEqual(remote([
                'import sys, os',
                'sys.stdout.write(os.environ["A_TEST_ENV"])'
            ], envar={'A_TEST_ENV': 'A_TEST_VALUE'}, stdout=PIPE),
            'A_TEST_VALUE'
        )

    def test_shell(self):
        from subprocess import PIPE
        from ..rpc import Remote
        remote = Remote('localhost')
        self.assertEqual(remote.shell([
                'echo "A_TEST_VALUE"',
            ]),
            'A_TEST_VALUE\n'
        )
