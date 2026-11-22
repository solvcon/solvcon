# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for Ctrl-C ending the pilot event loop.

The tests run the loop in-process. To check the pilot binary end to end, run
it with ``QT_QPA_PLATFORM=offscreen``, send it SIGINT, and expect exit status
130, both while it idles and while a console command runs.
"""

import os
import signal
import threading
import unittest

from solvcon import apputil, testing

try:
    from PySide6 import QtCore
    from solvcon.pilot.base import _gui_common
    from solvcon.pilot.panel import _console
except ImportError:
    QtCore = None


@testing.requires_pilot
@unittest.skipIf(os.name == 'nt', 'os.kill cannot raise SIGINT on Windows')
class InterruptExitsTC(unittest.TestCase):
    """The tests end only their own loop: QCoreApplication.exit would leave
    every later event loop of the pytest worker returning at once."""

    @staticmethod
    def _exec_with_sigint():
        """Return the status of a loop that gets SIGINT after 0.5 s.

        The status is -1 if the loop still runs after 2 s. The signal comes
        off the main thread, so only the wake timer of
        :func:`interrupt_exits` can deliver it while the loop idles.
        """
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(2000, lambda: loop.exit(-1))
        killer = threading.Timer(0.5, os.kill,
                                 args=(os.getpid(), signal.SIGINT))
        with _gui_common.interrupt_exits(stop=loop.exit):
            killer.start()
            ret = loop.exec()
        killer.join()
        return ret

    def test_sigint_exits_idle_loop(self):
        self.assertEqual(128 + signal.SIGINT, self._exec_with_sigint())

    def test_sigint_stops_console_command_then_loop(self):
        aenv = apputil.AppEnvironment('test_pilot_interrupt')
        self.addCleanup(apputil.stop_code, aenv)
        runs = []
        busy = ('import time\n'
                'end = time.monotonic() + 3\n'
                'while time.monotonic() < end:\n'
                '    pass\n')
        QtCore.QTimer.singleShot(
            0, lambda: runs.append(_console.run_cell(busy, aenv)))

        self.assertEqual(128 + signal.SIGINT, self._exec_with_sigint())
        [run] = runs
        self.assertTrue(run.raised)
        self.assertIn('KeyboardInterrupt', ''.join(t for _, t in run))

    def test_restores_previous_handler(self):
        previous = signal.getsignal(signal.SIGINT)
        with _gui_common.interrupt_exits():
            self.assertIsNot(previous, signal.getsignal(signal.SIGINT))
        self.assertIs(previous, signal.getsignal(signal.SIGINT))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
