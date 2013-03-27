# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2013 Yung-Yu Chen <yyc@solvcon.net>.
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

class TestMeshCase(TestCase):
    def test_empty_fields(self):
        from solvcon.case import MeshCase
        cse = MeshCase()
        # execution related.
        self.assertTrue(isinstance(cse.runhooks, list))
        self.assertEqual(cse.execution.time, 0.0)
        self.assertEqual(cse.execution.time_increment, 0.0)
        self.assertEqual(cse.execution.step_init, 0)
        self.assertEqual(cse.execution.step_current, None)
        self.assertEqual(cse.execution.steps_run, None)
        self.assertEqual(cse.execution.var, dict())
        self.assertEqual(cse.execution.varstep, None)
        # io related.
        self.assertEqual(cse.io.abspath, False)
        self.assertEqual(cse.io.basedir, None)
        self.assertEqual(cse.io.basefn, None)
        # condition related.
        self.assertEqual(cse.condition.mtrllist, list())
        # solver related.
        self.assertEqual(cse.solver.solvertype, None)
        self.assertEqual(cse.solver.solverobj, None)
        # logging.
        self.assertEqual(cse.log.time, dict())

    def test_abspath(self):
        import os
        from solvcon.case import MeshCase
        cse = MeshCase(basedir='.', abspath=True)
        path = os.path.abspath('.')
        self.assertEqual(cse.io.basedir, path)

    def test_init(self):
        from solvcon.testing import create_trivial_2d_blk
        from solvcon.domain import Domain
        from solvcon.solver import MeshSolver
        from solvcon.case import MeshCase
        blk = create_trivial_2d_blk()
        cse = MeshCase(basefn='meshcase', mesher=lambda *arg: blk,
            domaintype=Domain, solvertype=MeshSolver)
        cse.info.muted = True
        cse.init()

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
