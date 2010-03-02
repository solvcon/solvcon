# -*- coding: UTF-8 -*-

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

class TestOutpost(TestCase):
    def test_creation_termination(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from time import sleep
        try:
            from multiprocessing import Process
        except ImportError:
            from processing import Process
        from ..rpc import (DEFAULT_AUTHKEY, DEFAULT_SLEEP,
            pick_unused_port, Outpost, Footway)
        port = pick_unused_port()
        authkey = DEFAULT_AUTHKEY
        outpost = Outpost(publicaddress=('localhost', port), authkey=authkey)
        proc = Process(target=outpost.run)
        proc.start()
        sleep(DEFAULT_SLEEP)
        ftw = Footway(address=('localhost', port), authkey=authkey)
        # terminate.
        ftw.terminate()

    def test_muscle(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from time import sleep
        try:
            from multiprocessing import Process
            from multiprocessing.connection import Client
        except ImportError:
            from processing import Process
            from processing.connection import Client
        from ..rpc import (DEFAULT_AUTHKEY, DEFAULT_SLEEP,
            pick_unused_port, Outpost, Footway, Shadow)
        port = pick_unused_port()
        authkey = DEFAULT_AUTHKEY
        outpost = Outpost(publicaddress=('localhost', port), authkey=authkey)
        proc = Process(target=outpost.run)
        proc.start()
        sleep(DEFAULT_SLEEP)
        ftw = Footway(address=('localhost', port), authkey=authkey)
        # create worker and set muscle.
        pport = ftw.create()
        conn = Client(address=('localhost', pport), authkey=authkey)
        shadow = Shadow(connection=conn)
        muscle = Solver("solver0", "solver1")
        shadow.remote_setattr('muscle', muscle)
        # do something with the muscle.
        shadow.cmd.assert_msg('solver0')
        # terminate.
        ftw.terminate()

class TestFootway(TestCase):
    def test_build(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..rpc import DEFAULT_AUTHKEY, Footway
        authkey = DEFAULT_AUTHKEY
        port = Footway.build_outpost(address='localhost', authkey=authkey)
        ftw = Footway(address=('localhost', port), authkey=authkey)
        # terminate.
        ftw.terminate()

    def test_create(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        try:
            from multiprocessing.connection import Client
        except ImportError:
            from processing.connection import Client
        from ..rpc import DEFAULT_AUTHKEY, Footway, Shadow
        authkey = DEFAULT_AUTHKEY
        port = Footway.build_outpost(address='localhost', authkey=authkey)
        ftw = Footway(address=('localhost', port), authkey=authkey)
        # create worker and set muscle.
        pport = ftw.create()
        conn = Client(address=('localhost', pport), authkey=authkey)
        shadow = Shadow(connection=conn)
        muscle = Solver("solver0", "solver1")
        shadow.remote_setattr('muscle', muscle)
        # do something with the muscle.
        shadow.cmd.assert_msg('solver0')
        # terminate.
        ftw.terminate()
