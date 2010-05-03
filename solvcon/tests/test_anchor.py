from unittest import TestCase

from ..testing import TestingSolver
class CustomBlockSolver(TestingSolver):
    import os
    DEBUG_FILENAME_DEFAULT = os.devnull
    del os
del TestingSolver

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
