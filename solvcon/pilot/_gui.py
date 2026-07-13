# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Graphical-user interface code
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


from .. import apputil
from . import _pilot_core as _pcore
from . import airfoil

if _pcore.enable:
    from PySide6.QtGui import QAction
    from . import _gui_common
    from . import _mesh
    from . import _tree_panel
    from . import _oblique
    from . import _euler1d
    from . import _burgers1d
    from . import _svg_gui
    from . import _linear_wave
    from . import _canvas_gui
    from . import _painter_gui
    from . import _profiling
    from . import _agent_gui
    from . import _theme
    from . import _window_manager

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
        self._built = False
        self.mesh_sample_dialog = None
        self.gmsh_dialog = None
        self.svg_dialog = None
        self.tree_panel = None
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
        self.save_2d_canvas = None
        self.openprofiledata = None
        self.runprofiling = None
        self.agent = None
        self.theme_menu = None
        self.window_manager = None

    def __getattr__(self, name):
        return None if self._rmgr is None else getattr(self._rmgr, name)

    def launch(self, name="pilot", size=(1000, 600)):
        self.build(name=name, size=size)
        self._rmgr.show()
        return self._rmgr.exec()

    def build(self, name="pilot", size=(1000, 600)):
        """Assemble the window, features, and menu bar without the event
        loop, so the fully built bar can be exercised from a test.

        Idempotent: the bar is populated once, so a second call (for example
        from another test) returns the already-built manager unchanged.
        """
        if self._built:
            return self._rmgr
        self._rmgr = _pcore.RManager.instance
        self._rmgr.setUp()
        self._rmgr.windowTitle = name
        self._rmgr.resize(w=size[0], h=size[1])

        # Declare the Panels submenu first on the View menu; features place
        # their toggles under the "View/Panels" path.
        self._rmgr.menu_model.menu("View/Panels", weight=0)

        self.gmsh_dialog = _mesh.GmshFileDialog(mgr=self._rmgr)
        self.svg_dialog = _svg_gui.SVGFileDialog(mgr=self._rmgr)
        self.sample_mesh = _mesh.SampleMeshFeature(mgr=self._rmgr)
        self.mesh_style_status = _mesh.MeshStyleStatus(mgr=self._rmgr)
        self.tree_panel = _tree_panel.TreePanel(
            mgr=self._rmgr, style_status=self.mesh_style_status)
        self.oblique_shock = _oblique.ObliqueShockMesh(mgr=self._rmgr)
        self.oblique_solver = _oblique.ObliqueShockSolver(mgr=self._rmgr)
        self.naca4airfoil = airfoil.Naca4Airfoil(mgr=self._rmgr)
        self.mesh_sample_dialog = _mesh.SampleMeshDialog(
            mgr=self._rmgr, entries=self._mesh_sample_dialog_entries())
        self.eulerone = _euler1d.Euler1DApp(mgr=self._rmgr)
        self.burgers = _burgers1d.Burgers1DApp(mgr=self._rmgr)
        self.linear_wave = _linear_wave.LinearWave1DApp(mgr=self._rmgr)
        self.painter = _painter_gui.Painter(mgr=self._rmgr)
        self.canvas = _canvas_gui.Canvas(mgr=self._rmgr, painter=self.painter)
        self.save_2d_canvas = _canvas_gui.Save2DCanvasDialog(mgr=self._rmgr)
        self.openprofiledata = _profiling.Profiling(mgr=self._rmgr)
        self.runprofiling = _profiling.RunProfiling(mgr=self._rmgr)
        self.agent = _agent_gui.AgentPanel(mgr=self._rmgr)
        self.theme_menu = _theme.ThemeMenu(mgr=self._rmgr)
        self.window_manager = _window_manager.WindowManager(mgr=self._rmgr)
        self.populate_menu()
        self._seed_console_namespace()
        self._built = True
        return self._rmgr

    def _seed_console_namespace(self):
        """Curate the console namespace and greet with a banner."""
        appenv = apputil.get_current_appenv()
        banner = apputil.install_pilot_namespace(self._rmgr, appenv)
        self._rmgr.pycon.writeToHistory(banner)

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

        self.gmsh_dialog.populate_menu()
        self.svg_dialog.populate_menu()
        self.save_2d_canvas.populate_menu()
        self.tree_panel.populate_menu()
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
        self.agent.populate_menu()
        self.theme_menu.populate_menu()
        self.window_manager.populate_menu()

        # An explicit QuitRole lets macOS relocate Exit into the application
        # menu, so no platform special case is needed.
        wm.menu_model.place(
            "File",
            _gui_common.build_action(
                wm.mainWindow, "Exit", "Exit the application",
                lambda: wm.quit(), id="file.exit",
                menu_role=QAction.MenuRole.QuitRole),
            100)
        wm.menu_model.place(
            "View/Panels",
            _gui_common.build_action(
                wm.mainWindow, "Console", "Open / Close Console",
                wm.toggleConsole, id="panel.console",
                checkable=True, checked=True),
            30)


controller = _Controller()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
