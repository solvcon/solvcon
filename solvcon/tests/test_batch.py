# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestBatch(TestCase):
    def test_init(self):
        from ..case import BlockCase
        from ..batch import Batch
        case = BlockCase()

    def test_script(self):
        from ..case import BlockCase
        from ..batch import Batch
        case = BlockCase(rootdir='/tmp')
        sbm = Batch(case, arnname='arn')
        self.assertRaises(NotImplementedError, lambda: sbm.str_resource)
        self.assertRaises(NotImplementedError, lambda: sbm.str_jobname)
        self.assertRaises(NotImplementedError, lambda: sbm.str_output)
        self.assertRaises(NotImplementedError, lambda: sbm.str_shell)
        self.assertRaises(NotImplementedError, lambda: str(sbm))

class TestTorque(TestCase):
    SCRIPT = '''#!/bin/sh

#PBS -N arn
#PBS -j oe
#PBS -S /bin/sh
echo "Customized paths for job:"
. $HOME/.bashrc_path
export PYTHONPATH=/tmp:$PYTHONPATH
echo "Run @`date`:"
cd /tmp/arn
time %s run arn --runlevel %%d
echo "Finish @`date`."'''
    SCRIPT_FILE = '''#!/bin/sh

#PBS -N arn
#PBS -j oe
#PBS -S /bin/sh
echo "Customized paths for job:"
. $HOME/.bashrc_path
export PYTHONPATH=/tmp:$PYTHONPATH
echo "Run @`date`:"
cd /tmp/arn
time %s run arn --runlevel %d
echo "Finish @`date`."'''

    def test_script(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        from ..case import BlockCase
        from ..batch import batregy
        from ..conf import env
        case = BlockCase(rootdir='/tmp')
        sbm = batregy.Torque(case, arnname='arn')
        self.assertEqual(str(sbm), self.SCRIPT%env.get_entry_point())

    def test_tofile(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        import os, shutil
        from ..case import BlockCase
        from ..batch import batregy
        from ..conf import env
        case = BlockCase(rootdir='/tmp')
        msg = []
        case.info = lambda m: msg.append(m)
        sbm = batregy.Torque(case, arnname='arn')
        fnlist = sbm.tofile()
        for it in range(len(fnlist)):
            self.assertEqual(fnlist[it], '/tmp/arn/arn.pbs%d'%it)
            fn = fnlist[it]
            f = open(fn)
            self.assertEqual(f.read(),
                self.SCRIPT_FILE%(env.get_entry_point(), it))
            f.close()
        shutil.rmtree('/tmp/arn')
        self.assertFalse(os.path.exists('/tmp/arn'))

    def test_with_jobdir(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        import os, shutil
        from ..case import BlockCase
        from ..batch import batregy
        case = BlockCase(rootdir='/tmp')
        msg = []
        case.info = lambda m: msg.append(m)
        sbm = batregy.Torque(case, arnname='arn')
        if not os.path.exists('/tmp/arn'):
            os.makedirs('/tmp/arn')
        fnlist = sbm.tofile()
        for it in range(len(fnlist)):
            self.assertEqual(fnlist[it], '/tmp/arn/arn.pbs%d'%it)
        self.assertEqual(len(msg), 1)
        self.assertEqual(msg[0], 'Job directory was there: /tmp/arn\n')
        shutil.rmtree('/tmp/arn')
        self.assertFalse(os.path.exists('/tmp/arn'))

    def test_without_jobdir(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        import os, shutil
        from ..case import BlockCase
        from ..batch import batregy
        case = BlockCase(rootdir='/tmp')
        msg = []
        case.info = lambda m: msg.append(m)
        sbm = batregy.Torque(case, arnname='arn')
        if os.path.exists('/tmp/arn'):
            shutil.rmtree('/tmp/arn')
        fnlist = sbm.tofile()
        for it in range(len(fnlist)):
            self.assertEqual(fnlist[it], '/tmp/arn/arn.pbs%d'%it)
        self.assertEqual(len(msg), 0)
        shutil.rmtree('/tmp/arn')
        self.assertFalse(os.path.exists('/tmp/arn'))

    def test_with_empty(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        import os, shutil
        from ..case import BlockCase
        from ..batch import batregy
        case = BlockCase(rootdir='/tmp', empty_jobdir=True)
        msg = []
        case.info = lambda m: msg.append(m)
        sbm = batregy.Torque(case, arnname='arn')
        if not os.path.exists('/tmp/arn'):
            os.makedirs('/tmp/arn')
        fnlist = sbm.tofile()
        for it in range(len(fnlist)):
            self.assertEqual(fnlist[it], '/tmp/arn/arn.pbs%d'%it)
        self.assertEqual(len(msg), 2)
        self.assertEqual(msg[0], 'Job directory was there: /tmp/arn\n')
        self.assertEqual(msg[1], 'Delete all file in job directory.\n')
        shutil.rmtree('/tmp/arn')
        self.assertFalse(os.path.exists('/tmp/arn'))
