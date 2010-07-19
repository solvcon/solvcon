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
        from multiprocessing import Process
        from ..connection import pick_unused_port
        from ..rpc import DEFAULT_AUTHKEY, DEFAULT_SLEEP, Outpost, Footway
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
        from multiprocessing import Process
        from ..connection import pick_unused_port, Client
        from ..rpc import (DEFAULT_AUTHKEY, DEFAULT_SLEEP,
            Outpost, Footway, Shadow)
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

    def test_kill_remote(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..rpc import DEFAULT_AUTHKEY, Footway
        authkey = DEFAULT_AUTHKEY
        port = Footway.build_outpost(address='localhost', authkey=authkey)
        ftw = Footway(address=('localhost', port), authkey=authkey)
        pid = ftw.pid
        # terminate.
        ftw.terminate()
        # there should be no process to be killed remotely.
        msg = ftw.kill_remote()
        self.assertTrue('%d'%pid in msg)

    def test_create(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..connection import Client
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
