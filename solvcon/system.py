# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Runtime system code
"""


import builtins
import sys
import os
import argparse
import warnings

import solvcon
from . import apputil


__all__ = [
    'setup_process',
    'enter_main',
    'exec_code',
    'get_completions',
    'get_call_tip',
]


class ModmeshArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.exited = False
        self.exited_status = None
        self.exited_message = None

    def exit(self, status=0, message=None):
        self.exited = True
        self.exited_status = status
        self.exited_message = message
        return


def _parse_command_line(argv):
    parser = ModmeshArgumentParser(description="Pilot")
    parser.add_argument('--mode', dest='mode', action='store',
                        default='pilot',
                        choices=['pilot', 'python', 'pytest'],
                        help='mode selection (default = %(default)s)')
    # Unknown args (e.g., pytest flags after --mode=pytest) are forwarded to
    # the mode handler instead of erroring out.
    args, extra = parser.parse_known_args(argv[1:])
    args.extra = extra
    if parser.exited:
        args.exit = (parser.exited_status, parser.exited_message)
    else:
        args.exit = tuple()
    return args


def _run_pilot(argv=None):
    """Run the pilot application."""
    # The local importing pilot delays loading GUI/Qt code. GUI/Qt may be
    # unavailable in some execution modes and the module solvcon.pilot and
    # PySide6 should not be imported at module level.
    from . import pilot
    return pilot.launch()


def _run_pytest(extra_args=None):
    """Run the pytest harness against solvcon's tests directory.

    :param extra_args: Pytest options passed on the pilot command line
        (after ``--mode=pytest``). When non-empty, takes precedence over
        the ``PYTEST_OPTS`` environment variable, which is the path used
        by ``make run_pilot_pytest PYTEST_OPTS=...``.
    :type extra_args: list[str] or None
    :returns: Pytest exit code.
    :rtype: int
    """
    # Import pytest locally to avoid making it a dependency to the whole
    # solvcon.
    import pytest
    import shlex
    mmpath = os.path.join(os.path.dirname(solvcon.__file__), '..', 'tests')
    mmpath = os.path.abspath(mmpath)
    if extra_args:
        opts = list(extra_args)
    else:
        env_opts = os.environ.get('PYTEST_OPTS', '')
        opts = shlex.split(env_opts) if env_opts else ['-v', '-x']
    return pytest.main(opts + [mmpath])


def setup_process(argv):
    """Set up the runtime environment for the process."""
    # Install the namespace.
    builtins.solvcon = solvcon
    builtins.sc = solvcon

    if solvcon.HAS_PILOT:
        # Tolerate a broken pilot import so the GUI can still be entered;
        # warn instead of crashing the whole process.
        try:
            from . import pilot
            builtins.pilot = pilot
        except ImportError as e:
            warnings.warn("failed to import pilot: {}".format(e))


def enter_main(argv):
    args = _parse_command_line(argv)
    ret = 0
    if args.exit:
        ret = args.exit[0]
    elif args.extra and args.mode != 'pytest':
        # Extra args are only meaningful in pytest mode (forwarded to the
        # pytest runner). Reject them in other modes to surface typos.
        sys.stderr.write(
            'mode "{}" does not accept extra arguments: {}\n'.format(
                args.mode, ' '.join(args.extra)))
        ret = 2
    elif 'pilot' == args.mode:
        ret = _run_pilot(argv)
    elif 'pytest' == args.mode:
        ret = _run_pytest(args.extra)
    elif 'python' == args.mode:
        sys.stderr.write('mode "python" should not run in Python main')
        ret = 1
    else:
        sys.stderr.write('mode "{}" is not supported'.format(args.mode))
    return ret


def exec_code(code):
    # The persistent interpreter reports a user exception through its own
    # showtraceback/showsyntaxerror, which format the traceback against the
    # console input. Do not wrap it in the host call stack.
    return apputil.run_code(code)


def push_code(line):
    # Interactive per-line entry for the terminal console: feed one line and
    # report whether the statement is still incomplete, so the terminal can
    # switch between the primary and continuation prompts.
    return apputil.push_code(line)


def reset_console():
    # Abandon a partially entered statement, so the terminal can return to
    # the primary prompt without running the incomplete block.
    return apputil.reset_console()


def get_completions(text):
    try:
        return apputil.get_completions(text)
    except Exception as e:
        sys.stderr.write("get_completions error: {}\n".format(e))
        return []


def get_call_tip(text):
    try:
        return apputil.get_call_tip(text)
    except Exception as e:
        sys.stderr.write("get_call_tip error: {}\n".format(e))
        return ''

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
