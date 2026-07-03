# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Graphical-user interface code
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


import sys
import importlib

from . import _pilot_core as _pcore
from . import airfoil

if _pcore.enable:
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMenu
    from . import _mesh
    from . import _mesh_info
    from . import _oblique
    from . import _euler1d
    from . import _burgers1d
    from . import _svg_gui
    from . import _linear_wave
    from . import _canvas_gui
    from . import _painter_gui
    from . import _profiling

__all__ = [  # noqa: F822
    'controller',
    'launch',
]


def launch():
    return controller.launch()


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kw)
        return cls._instances[cls]


class _Controller(metaclass=_Singleton):
    def __init__(self):
        # Do not construct any Qt member objects before calling launch(), or
        # Windows may "exited with code -1073740791."
        self._rmgr = None
        self.panels_menu = None
        self.mesh_sample_dialog = None
        self.gmsh_dialog = None
        self.svg_dialog = None
        self.mesh_info = None
        self.sample_mesh = None
        self.mesh_style_status = None
        self.oblique_shock = None
        self.oblique_solver = None
        self.naca4airfoil = None
        self.eulerone = None
        self.burgers = None
        self.linear_wave = None
        self.painter = None
        self.canvas = None
        self.openprofiledata = None
        self.runprofiling = None

    def __getattr__(self, name):
        return None if self._rmgr is None else getattr(self._rmgr, name)

    def launch(self, name="pilot", size=(1000, 600)):
        self._rmgr = _pcore.RManager.instance
        self._rmgr.setUp()
        self._rmgr.windowTitle = name
        self._rmgr.resize(w=size[0], h=size[1])

        # Add the "Panels" submenu as the first item in the View menu.
        view = self._rmgr.viewMenu
        self.panels_menu = QMenu("Panels", self._rmgr.mainWindow)
        actions = view.actions()
        if actions:
            view.insertMenu(actions[0], self.panels_menu)
        else:
            view.addMenu(self.panels_menu)

        self.gmsh_dialog = _mesh.GmshFileDialog(mgr=self._rmgr)
        self.svg_dialog = _svg_gui.SVGFileDialog(mgr=self._rmgr)
        self.sample_mesh = _mesh.SampleMesh(mgr=self._rmgr)
        self.mesh_style_status = _mesh.MeshStyleStatus(mgr=self._rmgr)
        self.mesh_info = _mesh_info.MeshInfo(
            mgr=self._rmgr, menu=self.panels_menu,
            style_status=self.mesh_style_status)
        self.oblique_shock = _oblique.ObliqueShockMesh(mgr=self._rmgr)
        self.oblique_solver = _oblique.ObliqueShockSolver(mgr=self._rmgr)
        self.naca4airfoil = airfoil.Naca4Airfoil(mgr=self._rmgr)
        self.mesh_sample_dialog = _mesh.SampleMeshDialog(
            mgr=self._rmgr, entries=self._mesh_sample_dialog_entries())
        self.eulerone = _euler1d.Euler1DApp(mgr=self._rmgr)
        self.burgers = _burgers1d.Burgers1DApp(mgr=self._rmgr)
        self.linear_wave = _linear_wave.LinearWave1DApp(mgr=self._rmgr)
        self.painter = _painter_gui.Painter(mgr=self._rmgr,
                                            menu=self.panels_menu)
        self.canvas = _canvas_gui.Canvas(mgr=self._rmgr, painter=self.painter)
        self.openprofiledata = _profiling.Profiling(mgr=self._rmgr)
        self.runprofiling = _profiling.RunProfiling(mgr=self._rmgr)
        self.populate_menu()
        self._rmgr.show()
        return self._rmgr.exec()

    def _mesh_sample_dialog_entries(self):
        """Every example mesh as ``(category, label, tip, func)``, in menu
        order, gathered from the sample features for the sample dialog.  The
        features stay live so the dialog can invoke their bound methods.
        """
        return (self.sample_mesh.mesh_sample_dialog_entries()
                + self.oblique_shock.mesh_sample_dialog_entries()
                + self.naca4airfoil.mesh_sample_dialog_entries())

    def populate_menu(self):
        wm = self._rmgr

        def _addAction(menu, text, tip, func, checkable=False, checked=False):
            act = QAction(text, wm.mainWindow)
            act.setStatusTip(tip)
            act.setCheckable(checkable)
            if checkable:
                act.setChecked(checked)
            if callable(func):
                act.triggered.connect(lambda *a: func())
            elif func:
                modname, funcname = func.rsplit('.', maxsplit=1)
                mod = importlib.import_module(modname)
                func = getattr(mod, funcname)
                act.triggered.connect(lambda *a: func())
            menu.addAction(act)

        self.gmsh_dialog.populate_menu()
        self.svg_dialog.populate_menu()
        self.mesh_info.populate_menu()
        self.painter.populate_menu()
        self.mesh_sample_dialog.populate_menu()
        self.mesh_style_status.populate_menu()
        self.oblique_solver.populate_menu()
        self.eulerone.populate_menu()
        self.burgers.populate_menu()
        self.linear_wave.populate_menu()
        self.canvas.populate_menu()
        self.openprofiledata.populate_menu()
        self.runprofiling.populate_menu()

        if sys.platform != 'darwin':
            _addAction(
                menu=wm.fileMenu,
                text="Exit",
                tip="Exit the application",
                func=lambda: wm.quit(),
            )

        _addAction(
            menu=wm.windowMenu,
            text="Console",
            tip="Open / Close Console",
            func=wm.toggleConsole,
            checkable=True,
            checked=True,
        )


controller = _Controller()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
