from unittest import TestCase

class TestOnedim(TestCase):
    dnx = 101
    neq = 2

    def _get_solver(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        self.xgrid = empty(self.dnx, dtype=env.fpdtype)
        self.xmtrl = empty(self.dnx, dtype='int32')
        return OnedimSolver(self.xgrid, self.xmtrl,
            fpdtype=env.fpdtype, neq=self.neq)

class TestExeinfo(TestOnedim):
    def test_exeinfo(self):
        from ..onedim import OnedimSolverExeinfo
        einfo = OnedimSolverExeinfo()
        self.assertEqual(str(einfo), '''type execution
    integer*4 :: dnx = 0
    integer*4 :: neq = 0
    integer*4 :: nmtrl = 0
    integer*4 :: dnstep = 0
    real*8 :: time = 0.0
    real*8 :: time_increment = 0.0
    real*8 :: maxcfl = 0.0
end type execution'''
        )

class TestInit(TestOnedim):
    def test_success(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        xgrid = empty(101, dtype=env.fpdtype)
        xmtrl = empty(101, dtype='int32')
        OnedimSolver(xgrid, xmtrl, fpdtype=env.fpdtype, neq=0)

    def _xgridshape(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        xgrid = empty((101, 1), dtype=env.fpdtype)
        xmtrl = empty((101, 1), dtype='int32')
        OnedimSolver(xgrid, xmtrl, fpdtype=env.fpdtype, neq=0)
    def test_xgridshape(self):
        self.assertRaises(AssertionError, self._xgridshape)

    def _xgridlen(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        xgrid = empty(100, dtype=env.fpdtype)
        xmtrl = empty(100, dtype='int32')
        OnedimSolver(xgrid, xmtrl, fpdtype=env.fpdtype, neq=0)
    def test_xgridlen(self):
        self.assertRaises(AssertionError, self._xgridlen)

    def _xmtrllen(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        xgrid = empty(101, dtype=env.fpdtype)
        xmtrl = empty(100, dtype='int32')
        OnedimSolver(xgrid, xmtrl, fpdtype=env.fpdtype, neq=0)
    def test_xmtrllen(self):
        self.assertRaises(AssertionError, self._xmtrllen)

    def test_xmidlen(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        xgrid = empty(101, dtype=env.fpdtype)
        xmtrl = empty(101, dtype='int32')
        svr = OnedimSolver(xgrid, xmtrl, fpdtype=env.fpdtype, neq=0)
        self.assertEqual(svr.xmid.shape[0], xgrid.shape[0])
        # TODO: check for the midpoint is really the midpoint.

class TestGrid(TestOnedim):
    def test_xgrid(self):
        svr = self._get_solver()
        self.assertNotEqual(id(svr.xgrid), id(self.xgrid))

    def test_xmesh(self):
        svr = self._get_solver()
        self.assertNotEqual(id(svr.xmtrl), id(self.xmtrl))

class TestSolution(TestOnedim):
    def _get_solver(self):
        from numpy import empty
        from ...conf import env
        from ..onedim import OnedimSolver
        xgrid = empty(self.dnx, dtype=env.fpdtype)
        xmtrl = empty(self.dnx, dtype='int32')
        return OnedimSolver(xgrid, xmtrl, fpdtype=env.fpdtype, neq=self.neq)

    def test_solshape(self):
        svr = self._get_solver()
        self.assertEqual(len(svr.sol.shape), 2)
        self.assertEqual(svr.sol.shape[0], self.dnx)
        self.assertEqual(svr.sol.shape[1], self.neq)

    def test_dsolshape(self):
        svr = self._get_solver()
        self.assertEqual(len(svr.dsol.shape), 2)
        self.assertEqual(svr.dsol.shape[0], self.dnx)
        self.assertEqual(svr.dsol.shape[1], self.neq)

    def test_cflshape(self):
        svr = self._get_solver()
        self.assertEqual(len(svr.cfl.shape), 1)
        self.assertEqual(svr.dsol.shape[0], self.dnx)
