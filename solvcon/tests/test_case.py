# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2012 Yung-Yu Chen <yyc@solvcon.net>.
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
from ..anchor import VtkAnchor

class TestArrangement(TestCase):
    def test_arrangement_registry(self):
        from ..case import BaseCase, BlockCase
        self.assertNotEqual(
            id(BaseCase.arrangements),
            id(BlockCase.arrangements)
        )

class TestCaseInfo(TestCase):
    def test_CaseInfo(self):
        from ..case import CaseInfo
        tdic = CaseInfo(_defdict={
            'lay1_1': '1_1',
            'lay1_2.lay2_1': '1_2/2_1',
            'lay1_2.lay2_2': '1_2/2_2',
            'lay1_3.': '1_3/',
        })
        self.assertEqual(tdic.lay1_1, '1_1')
        self.assertEqual(tdic.lay1_2.lay2_1, '1_2/2_1')
        self.assertEqual(tdic.lay1_2.lay2_2, '1_2/2_2')
        self.assertEqual(len(tdic.lay1_3), 0)

class TestBaseCase(TestCase):
    def test_empty_fields(self):
        from ..case import BaseCase
        case = BaseCase()
        # execution related.
        self.assertTrue(isinstance(case.runhooks, list))
        self.assertEqual(case.execution.time, 0.0)
        self.assertEqual(case.execution.time_increment, 0.0)
        self.assertEqual(case.execution.step_init, 0)
        self.assertEqual(case.execution.step_current, None)
        self.assertEqual(case.execution.steps_run, None)
        self.assertEqual(case.execution.neq, 0)
        self.assertEqual(case.execution.var, dict())
        self.assertEqual(case.execution.varstep, None)
        # io related.
        self.assertEqual(case.io.abspath, False)
        self.assertEqual(case.io.basedir, None)
        self.assertEqual(case.io.basefn, None)
        # condition related.
        self.assertEqual(case.condition.mtrllist, list())
        # solver related.
        self.assertEqual(case.solver.solvertype, None)
        self.assertEqual(case.solver.solverobj, None)
        # logging.
        self.assertEqual(case.log.time, dict())

    def test_abspath(self):
        import os
        from ..case import BaseCase
        case = BaseCase(basedir='.', abspath=True)
        path = os.path.abspath('.')
        self.assertEqual(case.io.basedir, path)

    def test_init(self):
        from ..case import BaseCase
        case = BaseCase()
        case.init()

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
