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

class TestSecureShell(TestCase):
    def test_python(self):
        from subprocess import PIPE
        from solvcon.rpc import SecureShell
        remote = SecureShell('localhost')
        self.assertEqual(remote([
                'import sys, os',
                'sys.stdout.write(os.environ["A_TEST_ENV"])'
            ], envar={'A_TEST_ENV': 'A_TEST_VALUE'}, stdout=PIPE),
            'A_TEST_VALUE'
        )

    def test_shell(self):
        from solvcon.rpc import SecureShell
        remote = SecureShell('localhost')
        self.assertEqual(remote.shell([
                'echo "A_TEST_VALUE"',
            ]),
            'A_TEST_VALUE\n'
        )

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
