# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import importlib.util
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

_STUBBED = ("setuptools", "setuptools.command", "setuptools.command.build_ext")


class _Extension:

    def __init__(self, name, *args, **kw):
        self.name = name


class _BuildExt:
    """Stand-in for setuptools' build_ext.

    Only the hooks cmake_build_ext actually reaches are provided; each one
    records or returns what the real command would.
    """

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


def _load_setup():
    # setup.py is not importable as a package module, and importing it runs
    # only definitions: main() is guarded by __name__ == '__main__'.
    # setuptools is stubbed so this works without a build backend installed.
    setuptools = types.ModuleType("setuptools")
    setuptools.Extension = _Extension
    command = types.ModuleType("setuptools.command")
    build_ext = types.ModuleType("setuptools.command.build_ext")
    build_ext.build_ext = _BuildExt
    command.build_ext = build_ext
    setuptools.command = command
    stubs = dict(zip(_STUBBED, (setuptools, command, build_ext)))
    # Restore by key rather than mock.patch.dict, which rebuilds sys.modules
    # wholesale and would evict anything imported inside the block.
    saved = {name: sys.modules.get(name) for name in _STUBBED}
    sys.modules.update(stubs)
    try:
        path = pathlib.Path(__file__).resolve().parent.parent / "setup.py"
        spec = importlib.util.spec_from_file_location("solvcon_setup", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


class SplitCommandArgsTC(unittest.TestCase):

    def setUp(self):
        self.setup = _load_setup()

    def test_posix_splitting_honours_quotes(self):
        with mock.patch.object(os, "name", "posix"):
            self.assertEqual(
                self.setup.split_command_args('-j4 -DFOO="a b"'),
                ["-j4", "-DFOO=a b"])

    def test_windows_path_keeps_its_backslashes(self):
        # The whole point of the nt branch: shlex would otherwise eat the
        # backslashes of a Windows path as escape characters.
        with mock.patch.object(os, "name", "nt"):
            self.assertEqual(
                self.setup.split_command_args(
                    r"-DCMAKE_PREFIX_PATH=C:\scdv\usr -DBUILD_QT=OFF"),
                [r"-DCMAKE_PREFIX_PATH=C:\scdv\usr", "-DBUILD_QT=OFF"])

    def test_windows_path_may_contain_an_apostrophe(self):
        # cmd.exe has no single-quote quoting, so an apostrophe is an
        # ordinary character rather than an unterminated quote.
        with mock.patch.object(os, "name", "nt"):
            self.assertEqual(
                self.setup.split_command_args(r"-DP=C:\Users\O'Brien -DQ=1"),
                [r"-DP=C:\Users\O'Brien", "-DQ=1"])

    def test_hash_is_an_argument_not_a_comment(self):
        for name in ("posix", "nt"):
            with self.subTest(os_name=name):
                with mock.patch.object(os, "name", name):
                    self.assertEqual(
                        self.setup.split_command_args("-DX=a#b -DY=1"),
                        ["-DX=a#b", "-DY=1"])

    def test_empty_value_yields_no_arguments(self):
        for name in ("posix", "nt"):
            with self.subTest(os_name=name):
                with mock.patch.object(os, "name", name):
                    self.assertEqual(self.setup.split_command_args("   "), [])


class BuildCmakeTC(unittest.TestCase):
    """The argv cmake_build_ext hands to cmake, without running one."""

    def setUp(self):
        self.setup = _load_setup()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _run(self, cmake_args="", make_args="", debug=False):
        cmd = self.setup.cmake_build_ext.__new__(self.setup.cmake_build_ext)
        cmd.build_temp = os.path.join(self.tmp.name, "build")
        cmd.cmake_args = cmake_args
        cmd.make_args = make_args
        cmd.debug = debug
        cmd.get_ext_fullpath = lambda name: os.path.join(
            self.tmp.name, "out", name + ".pyd")
        ext = _Extension("_solvcon")
        with mock.patch.object(self.setup.subprocess, "run") as run:
            cmd.build_cmake(ext)
        self.calls = run.call_args_list
        return [call.args[0] for call in self.calls]

    def test_configure_and_build_are_both_invoked(self):
        configure, build = self._run()
        self.assertEqual(configure[:2], ["cmake", "-S"])
        self.assertEqual(build[:4], ["cmake", "--build", ".", "--config"])
        self.assertEqual(build[-2:], ["--target", "_solvcon"])

    def test_configure_pins_the_running_interpreter(self):
        configure, _ = self._run()
        self.assertIn("-DPYTHON_EXECUTABLE={}".format(sys.executable),
                      configure)

    def test_debug_selects_the_debug_config(self):
        configure, build = self._run(debug=True)
        self.assertIn("-DCMAKE_BUILD_TYPE=Debug", configure)
        self.assertEqual(build[build.index("--config") + 1], "Debug")

    def test_subprocess_is_not_run_through_a_shell(self):
        # The argv form is what keeps a path with spaces intact; a shell
        # string would re-split it.
        self._run()
        for call in self.calls:
            self.assertNotIn("shell", call.kwargs)
            self.assertTrue(call.kwargs["check"])

    def test_extra_make_args_go_after_the_separator(self):
        _, build = self._run(make_args="-j4")
        self.assertEqual(build[-2:], ["--", "-j4"])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
