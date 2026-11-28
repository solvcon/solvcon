# Copyright (c) 2019, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Graphical-user interface code
"""

# Use flake8 http://flake8.pycqa.org/en/latest/user/error-codes.html


from ... import apputil
from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _gui_common
    from .. import benchmark
    from .. import visual
    from .. import panel
    from .. import canvas
    from .. import painter as _painter
    from .. import track
    from . import _capture
    from . import _theme
    from . import _ui_state

__all__ = [  # noqa: F822
    'controller',
    'launch',
    'register_app',
]


def launch():
    return controller.launch()


def register_app(name, factory):
    return controller.register_app(name, factory)


def close_active_window(mgr):
    """Close the active sub-window, or Pilot when none is active."""
    subwin = mgr.mdiArea.activeSubWindow()
    if subwin is None:
        mgr.mainWindow.close()
    else:
        subwin.close()


def update_close_shortcut(action, mgr, subwin):
    """Yield Primary+W to an active MDI sub-window's close action."""
    if subwin is None:
        _gui_common.apply_shortcut(action, mgr=mgr)
    else:
        action.setShortcuts([])


class _Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kw):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kw)
        return cls._instances[cls]


class _AppLoader(object):
    """Stand for one app on the console shelf until :meth:`load` opens it.

    Reading the loader opens nothing, so the completer, which reads every
    attribute it lists, may list the apps.

    :param controller: The controller the app registered with.
    :param name: The name it registered as.
    """

    def __init__(self, controller, name):
        self._controller = controller
        #: The name the app registered as.
        self.name = name

    def __repr__(self):
        return f"<app {self.name!r}, which load() opens>"

    def load(self):
        """Open the app, as its menu item does, and return it."""
        app = self._controller.apps[self.name]
        app.open()
        return app


class _AppShelf(object):
    """The apps by name for the console: ``apps.obsrefl`` is the
    :class:`_AppLoader` of the app registered as ``obsrefl``.

    :param controller: The controller whose apps the shelf holds.
    """

    def __init__(self, controller):
        self._controller = controller

    def __dir__(self):
        return sorted(self._controller.apps)

    def __repr__(self):
        return f"apps: {', '.join(dir(self))}"

    def __getattr__(self, name):
        if name in self._controller.apps:
            return _AppLoader(self._controller, name)
        raise AttributeError(f"no app named {name!r}; the apps are "
                             f"{', '.join(dir(self))}")


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
        self.app_config = None
        self.apps = {}
        self._app_factories = {}
        self.app_shelf = _AppShelf(self)
        self.sample_mesh = None
        self.mesh_style_status = None
        self.painter = None
        self.canvas = None
        self.save_2d_canvas = None
        self.openprofiledata = None
        self.runprofiling = None
        self.benchmark = None
        self.mcap_panel = None
        self.console_panel = None
        self.theme_menu = None
        self.window_manager = None
        self.capture = None
        self.capture_widget = None
        self.ui_state = None

    def __getattr__(self, name):
        return None if self._rmgr is None else getattr(self._rmgr, name)

    @property
    def mgr(self):
        """The manager the features are built on, or None before build()."""
        return self._rmgr

    def register_app(self, name, factory):
        """Add an app to the pilot under ``name``.

        ``factory(controller)`` returns a feature with ``populate_menu()``.
        It runs in build(), or right away when the pilot is already built,
        and the feature lands in ``apps[name]`` and on :attr:`app_shelf`.  A
        name taken twice raises :class:`ValueError` rather than placing a
        second set of menu entries.
        """
        if name in self._app_factories:
            raise ValueError("app {!r} is already registered".format(name))
        self._app_factories[name] = factory
        if self._built:
            self._start_app(name)

    def _start_apps(self):
        for name in list(self._app_factories):
            self._start_app(name)

    def _start_app(self, name):
        app = self._app_factories[name](self)
        app.populate_menu()
        self.apps[name] = app

    def console_handles(self):
        """The console's app shelf with its banner lines."""
        handles = {'apps': self.app_shelf}
        entries = [
            ('apps.<name>.load()', 'open an app and return it; Tab completes'),
        ]
        return handles, entries

    def launch(self, name="pilot", size=(1000, 600)):
        self.build(name=name, size=size)
        self._rmgr.show()
        with _gui_common.interrupt_exits():
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

        self.gmsh_dialog = visual.GmshFileDialog(mgr=self._rmgr)
        self.svg_dialog = canvas.SVGFileDialog(mgr=self._rmgr)
        self.sample_mesh = visual.SampleMeshFeature(mgr=self._rmgr)
        self.mesh_style_status = visual.MeshStyleStatus(mgr=self._rmgr)
        self.tree_panel = panel.TreePanel(
            mgr=self._rmgr, style_status=self.mesh_style_status)
        self.mesh_sample_dialog = visual.SampleMeshDialog(
            mgr=self._rmgr, entries=self._mesh_sample_dialog_entries())
        self.painter = _painter.Painter(mgr=self._rmgr)
        self.canvas = canvas.Canvas(mgr=self._rmgr, painter=self.painter)
        self.save_2d_canvas = canvas.Save2DCanvasDialog(mgr=self._rmgr)
        self.openprofiledata = panel.Profiling(mgr=self._rmgr)
        self.runprofiling = panel.RunProfiling(mgr=self._rmgr)
        self.mcap_panel = track.McapPanel(mgr=self._rmgr)
        self.console_panel = panel.ConsolePanel(mgr=self._rmgr)
        self.theme_menu = _theme.ThemeMenu(mgr=self._rmgr)
        self.window_manager = panel.WindowManager(mgr=self._rmgr)
        self.capture = _capture.ScreenCapture(mgr=self._rmgr)
        self.capture_widget = panel.CaptureWidget(
            mgr=self._rmgr, capture=self.capture)
        self.app_config = panel.AppConfig(mgr=self._rmgr)
        self.ui_state = _ui_state.UiState(mgr=self._rmgr)
        self.dock_chrome = panel.DockChrome(mgr=self._rmgr)
        self.populate_menu()
        self._start_apps()
        self._seed_console_namespace()
        self._built = True
        return self._rmgr

    def _seed_console_namespace(self):
        """Curate the interpreter's namespace and tell the user how to list
        what it puts in scope."""
        handles, entries = self.console_handles()
        capture = self.capture.console_handles()
        handles.update(capture[0])
        entries.extend(capture[1])
        apputil.install_pilot_namespace(
            self._rmgr, apputil.get_current_appenv(),
            extra=(handles, entries))
        _gui_common.write_message(apputil.PILOT_CONSOLE_NOTE)

    def _open_benchmark(self):
        if self.benchmark is None:
            self.benchmark = benchmark.BenchmarkInspector()
            self._rmgr.mdiArea.addSubWindow(self.benchmark)
        self.benchmark.widget().show()
        self.benchmark.show()
        if self.benchmark.isMinimized():
            self.benchmark.showNormal()
        self._rmgr.mdiArea.setActiveSubWindow(self.benchmark)

    def _mesh_sample_dialog_entries(self):
        """Every example mesh as ``(category, label, tip, func)``, in menu
        order, gathered from the sample features for the sample dialog.  The
        features stay live so the dialog can invoke their bound methods.
        """
        return self.sample_mesh.mesh_sample_dialog_entries()

    def populate_menu(self):
        wm = self._rmgr

        self.gmsh_dialog.populate_menu()
        self.svg_dialog.populate_menu()
        self.save_2d_canvas.populate_menu()
        self.tree_panel.populate_menu()
        self.painter.populate_menu()
        self.mesh_sample_dialog.populate_menu()
        self.mesh_style_status.populate_menu()
        self.canvas.populate_menu()
        self.openprofiledata.populate_menu()
        self.runprofiling.populate_menu()
        self.mcap_panel.populate_menu()
        self.console_panel.populate_menu()
        self.theme_menu.populate_menu()
        self.window_manager.populate_menu()
        self.capture.populate_menu()
        self.capture_widget.populate_menu()
        self.app_config.populate_menu()

        close = _gui_common.build_action(
            wm.mainWindow, "Close",
            "Close the active sub-window, or Pilot when none is active",
            lambda: close_active_window(wm), id="window.close")
        wm.mdiArea.subWindowActivated.connect(
            lambda subwin: update_close_shortcut(close, wm, subwin))
        update_close_shortcut(close, wm, wm.mdiArea.activeSubWindow())
        wm.menu_model.place("File", close, 90)
        wm.menu_model.place(
            "File",
            _gui_common.build_action(
                wm.mainWindow, "Exit", "Exit the application",
                lambda: wm.quit(), id="file.exit"),
            100)
        wm.menu_model.place(
            "Profiling",
            _gui_common.build_action(
                wm.mainWindow, "Benchmark Inspector",
                "Compare operation kernels", self._open_benchmark,
                id="profiling.benchmark"),
            60)


controller = _Controller()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
