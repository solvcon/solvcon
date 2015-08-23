import unittest

from solvcon import testing

from .. import solver

class TestGasSolver(unittest.TestCase):
    def test_init(self):
        blk = testing.create_trivial_2d_blk()
        blk.clgrp.fill(0)
        blk.grpnames.append('blank')
        svr = solver.GasSolver(blk)

    def test_neq_2d(self):
        blk = testing.create_trivial_2d_blk()
        blk.clgrp.fill(0)
        blk.grpnames.append('blank')
        svr = solver.GasSolver(blk)
        self.assertEqual(4, svr.neq)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 tw=79:
