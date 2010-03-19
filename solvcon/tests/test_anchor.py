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

    def test_zeroi(self):
        import warnings
        from .. import anchor
        svr = CustomBlockSolver(self._get_block(), neq=self.neq)
        svr.runanchors.append(anchor.ZeroIAnchor(svr))
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
        svr.runanchors.append(anchor.RuntimeStatAnchor(svr,
            reports=['time', 'mem', 'loadavg', 'cpu']
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
