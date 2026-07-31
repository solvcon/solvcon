# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Dock panel that runs the 2D Euler oblique-shock solver and draws a
selected solution field as a flat color map.

The panel mirrors the mesh information panel: a feature toggled from the
View "Panels" submenu owns a control widget.  Here the controls set the free
stream and mesh, open or close the domain viewer sub-window, start / pause /
step the march, and pick which derived field (density, velocity, pressure,
Mach, ...) the viewer colors.
"""

import numpy as np

from PySide6.QtCore import Qt, QTimer, QObject, QEvent
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                               QDockWidget, QComboBox, QDoubleSpinBox,
                               QSpinBox, QPushButton, QTreeWidget,
                               QTreeWidgetItem, QFrame)

from ... import core
from ...multidim.euler import oblique
from . import _field_render
from ..base import _gui_common

__all__ = [  # noqa: F822
    'SolutionInfo',
]


class _SubWindowCloseFilter(QObject):
    """Report a watched sub-window's close synchronously.

    A ``QMdiSubWindow`` has no close signal, so this filter is the only way to
    stop the march before Qt frees the viewer.
    """

    def __init__(self, on_close, parent):
        super().__init__(parent)
        self._on_close = on_close

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Close:
            self._on_close()
        return False


class SolutionPanel(QWidget):
    """Widget with the solver controls and a live solution-field readout."""

    #: Derived scalar fields the viewer can color, in display order.
    FIELDS = ('density', 'velocity-x', 'velocity-y', 'speed',
              'pressure', 'mach', 'energy')
    #: Mesh flavors offered by :class:`~solvcon.multidim.euler.oblique`.
    CELL_TYPES = ('quad', 'triangle', 'unstructured')

    def __init__(self, parent=None):
        super().__init__(parent)
        # Owner-supplied callbacks that drive the solver from the controls.
        self.viewer_toggled = None
        self.start_requested = None
        self.pause_toggled = None
        self.step_requested = None
        self.field_changed = None
        self._build_controls()
        self._build_status()

    def _build_controls(self):
        """Lay out the free-stream / mesh inputs and the run buttons."""
        self._gamma = self._spin(1.4, 1.01, 3.0, 0.01, 3)
        self._density = self._spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._pressure = self._spin(1.0, 1e-3, 1e6, 0.1, 3)
        self._mach = self._spin(3.0, 1.1, 20.0, 0.1, 3)
        self._angle = self._spin(10.0, 0.5, 45.0, 0.5, 2)
        self._dt = self._spin(2e-3, 1e-6, 1.0, 1e-3, 6)
        self._steps = QSpinBox()
        self._steps.setRange(1, 1000)
        self._steps.setValue(5)
        self._cell_type = QComboBox()
        self._cell_type.addItems(self.CELL_TYPES)
        self._field = QComboBox()
        self._field.addItems(self.FIELDS)
        self._field.currentTextChanged.connect(self._on_field_changed)

        form = QFormLayout()
        form.addRow("gamma", self._gamma)
        form.addRow("density", self._density)
        form.addRow("pressure", self._pressure)
        form.addRow("mach", self._mach)
        form.addRow("angle (deg)", self._angle)
        form.addRow("dt", self._dt)
        form.addRow("steps/frame", self._steps)
        form.addRow("cell type", self._cell_type)
        form.addRow("field", self._field)

        # Opens and closes the one domain viewer the run buttons draw into.
        self._viewer_btn = QPushButton("Open viewer")
        self._viewer_btn.setCheckable(True)
        self._viewer_btn.toggled.connect(self._on_viewer_toggled)

        self._start = QPushButton("Start")
        self._start.clicked.connect(self._on_start_clicked)
        self._pause = QPushButton("Pause")
        self._pause.setCheckable(True)
        self._pause.toggled.connect(self._on_pause_toggled)
        self._step = QPushButton("Step")
        self._step.clicked.connect(self._on_step_clicked)
        buttons = QHBoxLayout()
        buttons.addWidget(self._start)
        buttons.addWidget(self._pause)
        buttons.addWidget(self._step)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.addLayout(form)
        self._layout.addWidget(self._viewer_btn)
        self._layout.addLayout(buttons)

    def _build_status(self):
        """Add the read-only step / value-range tree below the controls."""
        self._tree = QTreeWidget()
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        self._tree.setFrameShape(QFrame.NoFrame)
        self._layout.addWidget(self._tree)
        self.set_status(None, None, None)

    @staticmethod
    def _spin(value, low, high, step, decimals):
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def params(self):
        """Collect the current control values for the solver driver."""
        return dict(gamma=self._gamma.value(),
                    density=self._density.value(),
                    pressure=self._pressure.value(),
                    mach=self._mach.value(),
                    angle=self._angle.value(),
                    time_increment=self._dt.value(),
                    cell_type=self._cell_type.currentText())

    def field(self):
        return self._field.currentText()

    def steps_per_frame(self):
        return self._steps.value()

    def set_paused(self, paused):
        """Reflect the run state in the Pause button without re-firing it."""
        self._pause.blockSignals(True)
        self._pause.setChecked(paused)
        self._pause.setText("Resume" if paused else "Pause")
        self._pause.blockSignals(False)

    def set_viewer_open(self, open_):
        """Reflect the viewer state in its button without re-firing it."""
        self._viewer_btn.blockSignals(True)
        self._viewer_btn.setChecked(open_)
        self._viewer_btn.setText("Close viewer" if open_ else "Open viewer")
        self._viewer_btn.blockSignals(False)

    def set_status(self, step, vmin, vmax):
        """Show the marched step count and the drawn field's value range."""
        self._tree.clear()
        if step is None:
            QTreeWidgetItem(self._tree, ["not started"])
            return
        QTreeWidgetItem(self._tree, [f"step: {step}"])
        QTreeWidgetItem(self._tree, [f"field: {self.field()}"])
        QTreeWidgetItem(self._tree, [f"min: {vmin:.4g}"])
        QTreeWidgetItem(self._tree, [f"max: {vmax:.4g}"])

    @staticmethod
    def compute_field(name, cons, gamma, ndim):
        """Derive the named scalar field from the conserved variables.

        ``cons`` is the order-0 solution ``[ncell, neq]`` over the body cells
        -- density, the ``ndim`` momentum components, then total energy -- and
        ``gamma`` the matching per-cell ratio of specific heats.  Pressure
        follows the ideal-gas relation and Mach divides the speed by the local
        speed of sound.
        """
        rho = cons[:, 0]
        energy = cons[:, 1 + ndim]
        if name == 'density':
            return rho
        if name == 'energy':
            return energy
        vel = cons[:, 1:1 + ndim] / rho[:, None]
        if name == 'velocity-x':
            return vel[:, 0]
        if name == 'velocity-y':
            return vel[:, 1]
        speed2 = (vel ** 2).sum(axis=1)
        if name == 'speed':
            return np.sqrt(speed2)
        pressure = (gamma - 1.0) * (energy - 0.5 * rho * speed2)
        if name == 'pressure':
            return pressure
        if name == 'mach':
            return np.sqrt(speed2) / np.sqrt(gamma * pressure / rho)
        raise ValueError(f"unknown field '{name}'")

    @classmethod
    def solver_field(cls, svr, name):
        """Return the named field over ``svr``'s body (non-ghost) cells."""
        ng = svr.ngstcell
        return cls.compute_field(name, svr.so0n.ndarray[ng:],
                                 svr.gamma.ndarray[ng:], svr.ndim)

    def _on_viewer_toggled(self, open_):
        self._viewer_btn.setText("Close viewer" if open_ else "Open viewer")
        if self.viewer_toggled is not None:
            self.viewer_toggled(open_)

    def _on_start_clicked(self):
        if self.start_requested is not None:
            self.start_requested()

    def _on_pause_toggled(self, paused):
        self._pause.setText("Resume" if paused else "Pause")
        if self.pause_toggled is not None:
            self.pause_toggled(paused)

    def _on_step_clicked(self):
        if self.step_requested is not None:
            self.step_requested()

    def _on_field_changed(self, name):
        if self.field_changed is not None:
            self.field_changed(name)


class SolutionInfo(_gui_common.PilotFeature):
    """Euler solver panel, toggled from the View "Panels" submenu.

    The panel owns one domain viewer sub-window and one solver run.  The viewer
    control opens and closes the sub-window; starting marches the driver into
    it on a timer; closing it stops the march.
    """

    #: Stop the timer-driven march after this many steps.
    MAX_STEPS = 2000
    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Fired after a run sets the viewer mesh, so the inspector can refresh.
        self.viewer_updated = None
        self._action = None
        self._dock = None
        self._panel = None
        self._session = None
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        # Held for the panel's lifetime: a throwaway mdiArea wrapper is
        # garbage-collected right after use, invalidating any sub-window handle
        # taken through it.
        self._mdi = None

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Euler solver", "Toggle the Euler solver panel",
            None, id="panel.euler_solver", weight=20, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        if checked:
            self._ensure_panel()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        if self._panel is not None:
            return
        self._panel = SolutionPanel()
        self._panel.viewer_toggled = self._on_viewer
        self._panel.start_requested = self._on_start
        self._panel.pause_toggled = self._on_pause
        self._panel.step_requested = self._on_step
        self._panel.field_changed = self._on_field
        self._dock = QDockWidget("euler solver")
        self._dock.setWidget(self._panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea, self._dock)
        self._dock.visibilityChanged.connect(self._action.setChecked)

    def _on_viewer(self, open_):
        if open_:
            self._open_viewer()
        else:
            self._close_viewer()

    def _open_viewer(self):
        """Open the domain viewer sub-window if it is not already open."""
        if self._viewer is not None:
            return
        self._viewer = self._mgr.add3DWidget()
        self._viewer.showAxis(True)
        # Delete the sub-window on close and watch for that close, so a close
        # from any source stops the march before Qt frees the viewer.
        if self._mdi is None:
            self._mdi = self._mgr.mdiArea
        self._subwin = self._mdi.activeSubWindow()
        if self._subwin is not None:
            self._subwin.setAttribute(Qt.WA_DeleteOnClose, True)
            self._close_filter = _SubWindowCloseFilter(
                self._on_viewer_closed, self._subwin)
            self._subwin.installEventFilter(self._close_filter)
        self._panel.set_viewer_open(True)

    def _close_viewer(self):
        """Close the viewer sub-window; its close event stops the run."""
        if self._subwin is not None:
            self._subwin.close()
        else:
            self._on_viewer_closed()

    def _on_viewer_closed(self):
        # Reached from the sub-window's close event; stop the run and drop the
        # viewer before Qt frees it.
        self._stop_timer()
        self._viewer = None
        self._subwin = None
        self._close_filter = None
        self._panel.set_viewer_open(False)

    def _viewer_alive(self):
        """True while the viewer is open; the close filter clears it."""
        return self._viewer is not None

    def _on_start(self):
        """(Re)build the driver from the controls and march into the viewer,
        opening the viewer sub-window first if it was closed."""
        self._stop_timer()
        self._open_viewer()
        params = self._panel.params()
        shock = oblique.ObliqueShock()
        shock.build_constant(gamma=params['gamma'],
                             density=params['density'],
                             pressure=params['pressure'],
                             mach=params['mach'],
                             angle=params['angle'])
        shock.build_numerical(cell_type=params['cell_type'],
                              time_increment=params['time_increment'])
        # Set the viewer mesh so the inspector can report it; reusing an open
        # viewer raises no activation, so nudge the inspector directly.
        if self._viewer is not None:
            self._viewer.updateMesh(shock.mesh)
            if self.viewer_updated is not None:
                self.viewer_updated()
        fan, counts = _field_render.cell_triangulation(shock.mesh)
        # updateColorField wants an indexed vertex soup; the fan already is
        # one, so pack its vertices once (the geometry is fixed for the run)
        # and index them sequentially.
        verts = fan.pack_array().ndarray.reshape(-1, 3)
        indices = np.arange(verts.shape[0], dtype='uint32').reshape(-1, 3)
        timer = QTimer()
        timer.timeout.connect(self._advance)
        self._session = dict(
            shock=shock, timer=timer, counts=counts,
            verts=core.SimpleArrayFloat32(array=verts),
            indices=core.SimpleArrayUint32(array=indices), step=0)
        self._panel.set_paused(False)
        self._draw_frame()
        timer.start(self.INTERVAL_MS)

    def _on_pause(self, paused):
        if self._session is None:
            return
        if paused:
            self._session['timer'].stop()
        elif self._viewer_alive():
            self._session['timer'].start(self.INTERVAL_MS)

    def _on_step(self):
        if self._session is not None and self._viewer_alive():
            self._march_frame()

    def _on_field(self, _name):
        if self._session is not None:
            self._draw_frame()

    def _advance(self):
        if not self._viewer_alive():
            self._stop_timer()
            return
        if self._session['step'] >= self.MAX_STEPS:
            self._session['timer'].stop()
            self._panel.set_paused(True)
            return
        self._march_frame()

    def _march_frame(self):
        session = self._session
        steps = self._panel.steps_per_frame()
        session['shock'].march(steps)
        session['step'] += steps
        self._draw_frame()

    def _draw_frame(self):
        if not self._viewer_alive():
            return
        session = self._session
        field = SolutionPanel.solver_field(session['shock'].svr,
                                           self._panel.field())
        vmin, vmax = float(field.min()), float(field.max())
        colors = _field_render.field_colors(field, session['counts'],
                                            vmin, vmax)
        self._viewer.updateColorField(
            session['verts'], core.SimpleArrayFloat32(array=colors),
            session['indices'])
        self._panel.set_status(session['step'], vmin, vmax)

    def _stop_timer(self):
        if self._session is not None:
            self._session['timer'].stop()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
