# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""A shock tube runs, is judged, and is read out without the window.

The setup, the session, the view model, and the presenter carry what the
window shows and import no Qt, so a script or a build without Qt runs a
shock tube and reads what the window would draw.  A child Python with
PySide6 made unimportable does just that; a module among them that took to
importing Qt, directly or through another module, fails it.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import solvcon

#: The modules that carry the run and what the window shows of it.
HEADLESS = ('_setup', '_session', '_viewmodel', '_presenter')
#: What they import of the app in turn: the solvers and the analysis.
BENEATH = ('_analytic', '_riemann', '_shocktube')

#: What the child runs: the folder to report into, then the modules.
CHILD = """
import importlib
import importlib.abc
import json
import os
import sys


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if 'PySide6' == name.split('.')[0]:
            raise ImportError(f"{name} is blocked")
        return None


sys.meta_path.insert(0, Block())
from solvcon import pilot
folder, names = sys.argv[1], sys.argv[2:]
mods = {name: importlib.import_module('solvcon.apps.shocktube.' + name)
        for name in names}
presenter, viewmodel = mods['_presenter'], mods['_viewmodel']
setup = mods['_setup'].ShockTubeSetup(ncoord=41)
sess = setup.session(max_steps=20, steps_per_chunk=5)
sess.run()
display = viewmodel.Display()
results = presenter.results(sess)
profile = presenter.profile_frame(sess, display)
xt = presenter.xt_frame(sess)
strip = presenter.strip_state(sess, 'finished')
presenter.ShockTubePresenter(None, None).show(sess, display)
report = dict(
    pilot=pilot.enable, errors=list(setup.replace(ncoord=40).check()),
    states=[row.label for row in results.states],
    npoint=len(profile.points[0]), nrow=len(xt.rows), step=strip.step,
    qt=sorted(name for name, module in sys.modules.items()
              if 'PySide6' == name.split('.')[0] and module is not None),
    loaded=sorted(name.rsplit('.', 1)[1] for name in sys.modules
                  if name.startswith('solvcon.apps.shocktube.')))
with open(os.path.join(folder, 'report.json'), 'w') as stream:
    json.dump(report, stream)
"""


class HeadlessTC(unittest.TestCase):

    def test_the_run_reads_out_without_qt(self):
        # The pilot builds the extension into its own binary, so a child
        # Python started from inside it has no extension file to import,
        # whatever `sys.executable` names.
        impl = solvcon.core._impl
        if getattr(impl, '__file__', None) is None:
            self.skipTest('the extension is built into the Pilot')
        paths = [os.path.dirname(os.path.dirname(solvcon.__file__))]
        if '_solvcon' == impl.__name__:
            paths.append(os.path.dirname(impl.__file__))
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(
            paths + [part for part in [env.get('PYTHONPATH')] if part])
        with tempfile.TemporaryDirectory() as folder:
            done = subprocess.run(
                [sys.executable, '-c', CHILD, folder, *HEADLESS], env=env,
                capture_output=True, text=True, timeout=300)
            self.assertEqual(0, done.returncode, done.stderr)
            with open(os.path.join(folder, 'report.json')) as stream:
                report = json.load(stream)
        # The child really ran without Qt, so what it built needed none.
        self.assertFalse(report['pilot'])
        self.assertEqual([], report['qt'])
        # Reading a run out pulls in no widget module of the app.
        self.assertEqual(sorted(HEADLESS + BENEATH), report['loaded'])
        self.assertEqual(['ncoord'], report['errors'])
        self.assertEqual(['p*', 'u*', 'rho* left', 'rho* right'],
                         report['states'])
        self.assertEqual(19, report['npoint'])
        self.assertEqual(5, report['nrow'])
        self.assertEqual(20, report['step'])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
