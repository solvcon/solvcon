# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import contextlib
import io
import threading
import unittest

import solvcon


class AppenvTC(unittest.TestCase):

    def setUp(self):
        self.envbak = solvcon.apputil.environ.copy()
        self.envbasenum = len(self.envbak)

    def tearDown(self):
        solvcon.apputil.environ.clear()
        solvcon.apputil.environ = self.envbak.copy()

    def test_anonymous(self):
        def _check(i, basenum):
            env = solvcon.apputil.get_appenv()
            self.assertEqual(f'anonymous{i}', env.name)
            self.assertEqual(env, solvcon.apputil.environ[f'anonymous{i}'])
            self.assertEqual(basenum + i + 1,
                             len(solvcon.apputil.environ))

        _check(0, basenum=self.envbasenum)
        _check(1, basenum=self.envbasenum)
        _check(2, basenum=self.envbasenum)
        _check(3, basenum=self.envbasenum)
        _check(4, basenum=self.envbasenum)
        _check(5, basenum=self.envbasenum)
        _check(6, basenum=self.envbasenum)
        _check(7, basenum=self.envbasenum)
        _check(8, basenum=self.envbasenum)
        _check(9, basenum=self.envbasenum)

        with self.assertRaisesRegex(
                ValueError, r'hit limit of anonymous environments \(10\)'):
            _check(10, basenum=self.envbasenum)

        # Try to reset the environment dictionary.
        solvcon.apputil.environ.clear()
        _check(0, basenum=0)


class RunCodeStdinTC(unittest.TestCase):
    """The embedded console has no interactive stdin, so a command that
    reads it must hit EOF at once rather than block the GUI thread."""

    def setUp(self):
        self.envbak = solvcon.apputil.environ.copy()

    def tearDown(self):
        solvcon.apputil.environ.clear()
        solvcon.apputil.environ = self.envbak.copy()

    def _run_with_timeout(self, source, timeout=10):
        env = solvcon.apputil.AppEnvironment('stdin-test')
        done = threading.Event()

        # Run on a daemon thread so a regression that hangs fails the
        # assertion instead of stalling the whole test process.
        def _run():
            with contextlib.redirect_stdout(io.StringIO()):
                env.run_code(source)
            done.set()
        threading.Thread(target=_run, daemon=True).start()
        self.assertTrue(done.wait(timeout=timeout),
                        "run_code blocked reading stdin")
        return env

    def test_help_does_not_hang(self):
        self._run_with_timeout('help()')

    def test_input_reads_eof(self):
        env = self._run_with_timeout(
            'got_eof = False\n'
            'try:\n'
            '    input()\n'
            'except EOFError:\n'
            '    got_eof = True\n')
        self.assertTrue(env.globals['got_eof'])

    def test_stdin_restored_after_run(self):
        saved = solvcon.apputil.sys.stdin
        self._run_with_timeout('pass')
        self.assertIs(solvcon.apputil.sys.stdin, saved)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
