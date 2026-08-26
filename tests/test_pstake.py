# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import contextlib
import io
import os
import types
import unittest
from unittest import mock

from solvcon import pstake


class ExternalCommandTC(unittest.TestCase):

    def test_arguments_with_spaces_stay_separate(self):
        command = pstake.ExternalCommand("tool", echo=False)
        output = io.StringIO()
        with mock.patch("solvcon.pstake.subprocess.check_call") as check:
            command("input with space.tex", "-o", "output with space.eps",
                    cmdout=output)
        check.assert_called_once_with(
            ["tool", "input with space.tex", "-o",
             "output with space.eps"], stdout=output, stderr=None)

    def test_image_tools_name_every_spelling(self):
        renderer = pstake.Pstricks(quiet=True)
        self.assertEqual(
            renderer.cmd_convert.candidates, ("magick", "convert"))
        self.assertEqual(renderer.cmd_gs.candidates, ("gs", "gswin64c"))

    def test_command_takes_the_first_installed_candidate(self):
        command = pstake.ExternalCommand(("magick", "convert"), echo=False)
        with mock.patch("solvcon.pstake.shutil.which",
                        side_effect=lambda name: "/usr/bin/convert"
                        if name == "convert" else None):
            self.assertEqual(command.command, "convert")

    def test_command_falls_back_when_nothing_is_installed(self):
        command = pstake.ExternalCommand(("gs", "gswin64c"), echo=False)
        with mock.patch("solvcon.pstake.shutil.which", return_value=None):
            self.assertEqual(command.command, "gs")
            self.assertIsNone(command.command_abspath)

    def test_echoed_command_is_quoted_for_the_platform(self):
        command = pstake.ExternalCommand("tool", echo=False)
        output = io.StringIO()
        with mock.patch("solvcon.pstake.subprocess.check_call"):
            with mock.patch("solvcon.pstake.os.name", "nt"):
                command("a b.tex", cmdout=output)
            with mock.patch("solvcon.pstake.os.name", "posix"):
                command("a b.tex", cmdout=output)
        windows, posix = output.getvalue().splitlines()
        self.assertEqual(windows, 'tool "a b.tex"')
        self.assertEqual(posix, "tool 'a b.tex'")

    def test_command_abspath_is_absolute(self):
        command = pstake.ExternalCommand("tool", echo=False)
        with mock.patch("solvcon.pstake.shutil.which",
                        return_value=os.path.join("rel", "tool")):
            self.assertTrue(os.path.isabs(command.command_abspath))

    def test_quiet_main_uses_platform_null_device(self):
        args = types.SimpleNamespace(
            quiet=True, cmdout=None, src="input.tex", dst=None,
            dstext="png", font=None, options=False, keep_tmp=False,
            tempdir=None)
        opened = mock.mock_open()
        patches = (
            mock.patch("solvcon.pstake.argparse.ArgumentParser.parse_args",
                       return_value=args),
            mock.patch("builtins.open", opened),
            mock.patch("solvcon.pstake.Pstricks"),
            mock.patch("solvcon.pstake.Filename"),
            mock.patch("solvcon.pstake.os.devnull", "<null-device>"),
        )
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            self.assertEqual(pstake.main(), 0)
        opened.assert_called_once_with("<null-device>", "a+")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
