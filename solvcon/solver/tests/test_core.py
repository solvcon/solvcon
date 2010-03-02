from unittest import TestCase

from ..core import BaseSolver

class TestSolver(BaseSolver):
    def bind(self):
        super(TestSolver, self).bind()
        self.val = 'bind'

class TestCore(TestCase):
    def test_base(self):
        bsvr = BaseSolver()
        self.assertEqual(getattr(bsvr, 'val', None), None)
        bsvr.bind()
        self.assertEqual(getattr(bsvr, 'val', None), None)

    def test_inheritance(self):
        svr = TestSolver()
        self.assertEqual(getattr(svr, 'val', None), None)
        svr.bind()
        self.assertEqual(svr.val, 'bind')

class TestFpdtype(TestCase):
    def test_fp(self):
        from ...dependency import pointer_of, str_of
        from ...conf import env
        bsvr = BaseSolver()
        self.assertEqual(bsvr.fpdtype, env.fpdtype)
        self.assertEqual(bsvr.fpdtypestr, str_of(env.fpdtype))
        self.assertEqual(bsvr.fpptr, pointer_of(env.fpdtype))
