from unittest import TestCase

from ..solver import BlockSolver
class CustomBlockSolver(BlockSolver):
    import os
    DEBUG_FILENAME_DEFAULT = os.devnull
    del os
del BlockSolver

class TestAnchor(TestCase):
    neq = 1
    time = 0.0
    time_increment = 1.0
    nsteps = 10

    @staticmethod
    def _get_block():
        from ..testing import get_blk_from_sample_neu
        return get_blk_from_sample_neu()

    @classmethod
    def _get_solver(cls, init=True):
        import warnings
        svr = CustomBlockSolver(cls._get_block(), neq=cls.neq, enable_mesg=True)
        if init:
            warnings.simplefilter("ignore")
            svr.bind()
            svr.init()
            warnings.resetwarnings()
        return svr

    def test_runwithanchor(self):
        import warnings
        from ..anchor import Anchor
        svr = CustomBlockSolver(self._get_block(),
            neq=self.neq, enable_mesg=True)
        svr.runanchors.append(Anchor(svr))
        warnings.simplefilter("ignore")
        svr.bind()
        svr.init()
        warnings.resetwarnings()
        svr.soln.fill(0.0)
        svr.dsoln.fill(0.0)
        # run.
        svr.march(self.time, self.time_increment, self.nsteps)
        svr.final()
