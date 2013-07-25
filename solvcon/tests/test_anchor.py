import os
from unittest import TestCase

from ..solver import BlockSolver

class CustomBlockSolver(BlockSolver):
    DEBUG_FILENAME_DEFAULT = os.devnull

    _interface_init_ = ['cecnd', 'cevol']

    def __init__(self, blk, *args, **kw):
        """
        @keyword neq: number of equations (variables).
        @type neq: int
        """
        from numpy import empty
        super(CustomBlockSolver, self).__init__(blk, *args, **kw)
        # data structure for C/FORTRAN.
        self.blk = blk
        # arrays.
        ndim = self.ndim
        ncell = self.ncell
        ngstcell = self.ngstcell
        ## solutions.
        neq = self.neq
        self.sol = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.soln = empty((ngstcell+ncell, neq), dtype=self.fpdtype)
        self.dsol = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        self.dsoln = empty((ngstcell+ncell, neq, ndim), dtype=self.fpdtype)
        ## metrics.
        self.cecnd = empty(
            (ngstcell+ncell, self.CLMFC+1, ndim), dtype=self.fpdtype)
        self.cevol = empty((ngstcell+ncell, self.CLMFC+1), dtype=self.fpdtype)

    def create_alg(self):
        from solvcon.parcel.fake._algorithm import FakeAlgorithm
        alg = FakeAlgorithm()
        alg.setup_mesh(self.blk)
        alg.setup_algorithm(self)
        return alg

    ##################################################
    # marching algorithm.
    ##################################################
    MMNAMES = list()
    MMNAMES.append('update')
    def update(self, worker=None):
        self.sol[:,:] = self.soln[:,:]
        self.dsol[:,:,:] = self.dsoln[:,:,:]

    MMNAMES.append('calcsoln')
    def calcsoln(self, worker=None):
        self.create_alg().calc_soln()

    MMNAMES.append('ibcsoln')
    def ibcsoln(self, worker=None):
        if worker: self.exchangeibc('soln', worker=worker)

    MMNAMES.append('calccfl')
    def calccfl(self, worker=None):
        self.marchret = -2.0

    MMNAMES.append('calcdsoln')
    def calcdsoln(self, worker=None):
        self.create_alg().calc_dsoln()

    MMNAMES.append('ibcdsoln')
    def ibcdsoln(self, worker=None):
        if worker: self.exchangeibc('dsoln', worker=worker)

class TestAnchor(TestCase):
    neq = 1
    time = 0.0
    time_increment = 1.0
    nsteps = 10

    @staticmethod
    def _get_block():
        from ..testing import get_blk_from_sample_neu
        return get_blk_from_sample_neu()

    def test_runwithanchor(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.Anchor(svr))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_assign_type(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.Anchor)
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_assign_name(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        self.assertRaises(ValueError,
            svr.runanchors.append, anchor.Anchor, name=1,
        )
        svr.runanchors.append(anchor.Anchor, name='name')
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_zeroi(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.FillAnchor(svr,
            keys=('soln', 'dsoln'), value=0.0,
        ))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_runtimestat(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.RuntimeStatAnchor(svr))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_marchstat(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.MarchStatAnchor(svr))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_tpoolstat(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.TpoolStatAnchor(svr))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()

    def test_runtimestat2(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.RuntimeStatAnchor(svr,
            cputotal=False,
        ))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()
