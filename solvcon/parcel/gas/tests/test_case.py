import unittest

from solvcon import testing

from .. import case

class TestGasCase(unittest.TestCase):
    def test_init(self):
        def mesher(*args, **kw):
            blk = testing.create_trivial_2d_blk()
            blk.clgrp.fill(0)
            blk.grpnames.append('blank')
            return blk
        cse = case.GasCase(mesher=mesher)

# vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4 tw=79:
