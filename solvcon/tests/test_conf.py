# -*- coding: UTF-8 -*-

import os
if 'SOLVCON_FPDTYPE' in os.environ:
    del os.environ['SOLVCON_FPDTYPE']
if 'SOLVCON_INTDTYPE' in os.environ:
    del os.environ['SOLVCON_INTDTYPE']
from unittest import TestCase

class ConfTest(TestCase):
    def test_default_fpdtype(self):
        import numpy as np
        from ..conf import env
        if 'SOLVCON_FPDTYPE' in os.environ:
            fpdtype = getattr(np, os.environ['SOLVCON_FPDTYPE'])
            self.assertEqual(env.fpdtype, fpdtype)

    def test_fpdtypes(self):
        import numpy as np
        from .. import conf
        os.environ['SOLVCON_FPDTYPE'] = 'float64'
        self.assertEqual(conf.Solvcon().fpdtype, np.float64)
        os.environ['SOLVCON_FPDTYPE'] = 'float32'
        self.assertEqual(conf.Solvcon().fpdtype, np.float32)
