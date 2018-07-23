# -*- coding: UTF-8 -*-
# Copyright (C) 2016 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.


import unittest
from unittest import TestCase

import numpy as np

from .. import dependency
dependency.import_module_may_fail('..march')
from .. import block
from ..testing import get_blk_from_oblique_neu, get_blk_from_sample_neu

class TestCreation(TestCase):

    def test_import(self):
        from libmarch import gas

    def test_constructor(self):
        svr = march.gas.Solver2D(
            get_blk_from_oblique_neu(),
            sigma0=3,
            time=0,
            time_increment=0.1,
            report_interval=1,
        )


class TestGasSolver(TestCase):

    def test_class_attributes(self):
        for cls in (march.gas.Solver2D, march.gas.Solver3D):
            self.assertEqual(getattr(cls, '_interface_init_'), ('cecnd',))
            self.assertEqual(getattr(cls, '_solution_array_'), tuple())
