from unittest import TestCase

class TestOnedim(TestCase):
    def test_dnx(self):
        from ..onedim import OnedimCase
        case = OnedimCase()
        self.assertEqual(case.execution.dnx, None)

    def test_init(self):
        from ..onedim import OnedimCase
        case = OnedimCase()
        self.assertFalse(case._have_init)
        self.assertRaises(NotImplementedError, case.init)

class TestHook(TestCase):
    def test_existence(self):
        from .. import onedim
        self.assertTrue(onedim.OnedimHook)
        self.assertTrue(onedim.Initializer)
        self.assertTrue(onedim.Calculator)
        self.assertTrue(onedim.Plotter)
        self.assertTrue(onedim.Movie)

    def test_type(self):
        from .. import onedim
        self.assertRaises(AssertionError, onedim.OnedimHook, None)

    def test_initializer(self):
        from ..core import Hook
        from ..onedim import OnedimCase, Initializer
        case = OnedimCase()
        hook = Initializer(case)
        self.assertRaises(NotImplementedError, hook.preloop)
        self.assertNotEqual(Initializer.preloop, Hook.preloop)
        self.assertEqual(Initializer.premarch, Hook.premarch)
        self.assertEqual(Initializer.postmarch, Hook.postmarch)
        self.assertEqual(Initializer.postloop, Hook.postloop)

    def test_calculator(self):
        from ..core import Hook
        from ..onedim import OnedimCase, Calculator
        case = OnedimCase()
        hook = Calculator(case)
        self.assertRaises(NotImplementedError, hook._calculate)
        self.assertEqual(Calculator.preloop, Hook.preloop)
        self.assertEqual(Calculator.premarch, Hook.premarch)
        self.assertEqual(Calculator.postmarch, Hook.postmarch)
        self.assertEqual(Calculator.postloop, Hook.postloop)

class PlotterTest(TestCase):
    def setUp(self):
        import matplotlib as mpl
        mpl.rcParams['backend'] = 'Agg'
        self._msg = ''

    def info(self, msg):
        self._msg += msg

    def assertInfo(self, msg):
        self.assertEqual(self._msg, msg)

class TestPlotter(PlotterTest):
    def test_mencoder(self):
        from ..onedim import OnedimCase, Plotter

    def test_property(self):
        from matplotlib.figure import Figure
        from ..onedim import OnedimCase, Plotter
        case = OnedimCase()
        hook = Plotter(case)
        self.assertInfo('')
        self.assertEqual(hook.imgfn, None)
        self.assertTrue(isinstance(hook.fig, Figure))
        self.assertEqual(hook.outdir, None)

    def test_property_withimg(self):
        import os
        from matplotlib.figure import Figure
        from ..onedim import OnedimCase, Plotter
        case = OnedimCase(basedir='.')
        case.info = self.info
        hook = Plotter(case, imgfn='imgfn')
        self.assertInfo('Unlink %s ...\n' % os.sep.join(['.', '*.png']))
        self.assertEqual(hook.imgfn, 'imgfn')
        self.assertTrue(isinstance(hook.fig, Figure))
        self.assertEqual(hook.outdir, '.')

    def test_flags_preset(self):
        from ..onedim import OnedimCase, Plotter
        case = OnedimCase()
        hook = Plotter(case, flags={'test1': True})
        self.assertEqual(hook.flag_test1, True)
        self.assertRaises(KeyError, lambda: hook.flag_test2)

    def test_flags_postset(self):
        from ..onedim import OnedimCase, Plotter
        case = OnedimCase()
        hook = Plotter(case)
        hook.flag_test3 = True
        self.assertEqual(hook.flag_test3, True)

    def test_methods(self):
        from ..core import Hook
        from ..onedim import Plotter
        self.assertTrue(callable(Plotter._reltext))
        self.assertTrue(callable(Plotter._sync_legend_linewidth))
        self.assertEqual(Plotter.preloop, Hook.preloop)
        self.assertEqual(Plotter.premarch, Hook.premarch)
        self.assertEqual(Plotter.postmarch, Hook.postmarch)
        self.assertEqual(Plotter.postloop, Hook.postloop)

class TestMovie(PlotterTest):
    def test_property(self):
        import os
        from ..onedim import OnedimCase, Movie
        case = OnedimCase(basedir='.', steps_run=10)
        case.info = self.info
        hook = Movie(case, imgfn='')
        self.assertInfo('')
        self.assertEqual(hook.fps, 10)
        self.assertEqual(hook.width, 800)
        self.assertEqual(hook.height, 600)
        self.assertEqual(hook.snapshots, list())
        self.assertEqual(hook.vcodec, 'msmpeg4v2')
        self.assertEqual(hook.imgtmpl, 'None__%01d')
        self.assertEqual(hook.snptmpl, None)

    def test_property_with_imgfn(self):
        import sys
        from nose.plugins.skip import SkipTest
        if sys.platform.startswith('win'): raise SkipTest
        import os
        from ..onedim import OnedimCase, Movie
        has_mencoder = False
        for dire in os.environ['PATH'].split(':') + ['.']:
            if os.path.exists(os.path.join(dire, 'mencoder')):
                has_mencoder = True
                break
        case = OnedimCase(basedir='.', steps_run=10)
        case.info = self.info
        if not has_mencoder:
            self.assertRaises(OSError, lambda: Movie(case, imgfn='imgfn'))
            raise SkipTest
        hook = Movie(case, imgfn='imgfn')
        self.assertInfo('Unlink %s ...\n' % os.sep.join(['.', '*.png']))
        self.assertEqual(hook.fps, 10)
        self.assertEqual(hook.width, 800)
        self.assertEqual(hook.height, 600)
        self.assertEqual(hook.snapshots, list())
        self.assertEqual(hook.vcodec, 'msmpeg4v2')
        self.assertEqual(hook.imgtmpl, 'None_imgfn_%01d')
        self.assertEqual(hook.snptmpl, None)

    def test_snapshots(self):
        from ..onedim import OnedimCase, Movie
        case = OnedimCase(steps_run=10)
        hook = Movie(case, snapshots=[1])
        self.assertEqual(hook.snapshots, [1])
        self.assertEqual(hook.snptmpl, 'None_1_snapshot_%01d')

    def test_methods(self):
        from ..core import Hook
        from ..onedim import Movie
        self.assertTrue(callable(Movie._locate_mencoder))
        self.assertTrue(callable(Movie._redraw))
        self.assertTrue(callable(Movie._encode_movie))
        self.assertEqual(Movie.preloop, Hook.preloop)
        self.assertEqual(Movie.premarch, Hook.premarch)
        self.assertEqual(Movie.postmarch, Hook.postmarch)
        self.assertEqual(Movie.postloop, Hook.postloop)
