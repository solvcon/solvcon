# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Dock panel showing the 2D world's entities and diagnostics as a tree."""

import json

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QDockWidget,
                               QButtonGroup, QRadioButton, QTreeWidget,
                               QTreeWidgetItem, QFrame)

from . import _gui_common

__all__ = [  # noqa: F822
    'EntityTreePanel',
]


class EntityTreeWidget(QWidget):
    """Widget that presents the entity tree inside the dock."""

    LEVELS = ("basic", "diagnostics")

    # A unique sentinel for "nothing rendered/fingerprinted yet", distinct
    # from the ``None`` key that marks the "No world loaded" render.
    _UNSET = object()

    def __init__(self, world=None, parent=None):
        super().__init__(parent)
        self._world = None
        self._fingerprint = self._UNSET
        self._cache = {}
        self._rendered_key = self._UNSET
        self._tree = QTreeWidget()
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        self._tree.setFrameShape(QFrame.NoFrame)

        self._levels = {}
        level_row = QHBoxLayout()
        level_row.setContentsMargins(4, 2, 4, 2)
        group = QButtonGroup(self)
        for level in self.LEVELS:
            button = QRadioButton(level)
            group.addButton(button)
            level_row.addWidget(button)
            self._levels[level] = button
        level_row.addStretch(1)
        self._levels["diagnostics"].setChecked(True)

        for button in self._levels.values():
            button.toggled.connect(self._on_level_toggled)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(level_row)
        layout.addWidget(self._tree)

        self.setLayout(layout)
        self.set_world(world)

    @classmethod
    def make_world_info(cls, state):
        """Build the geometry sections as ``(section, rows)`` groups."""
        shapes = state.get("shapes", [])
        sections = [
            ("Counts", [
                ["shape", str(len(shapes))],
                ["bare segment", str(len(state.get("segments", [])))],
                ["bare curve", str(len(state.get("curves", [])))],
                ["point", str(len(state.get("points", [])))],
            ]),
        ]

        if shapes:
            rows = [[f"shape {sh['id']}", sh["type"]] for sh in shapes]
            sections.append(("Shapes", rows))
        bounds = cls.world_bounds(state)

        if bounds is not None:
            lower_x, lower_y, upper_x, upper_y = bounds
            sections.append(("Bounding box", [
                ["x", f"[{lower_x:.4g}, {upper_x:.4g}]"],
                ["y", f"[{lower_y:.4g}, {upper_y:.4g}]"],
            ]))
        return sections

    @staticmethod
    def world_bounds(state):
        """Union the x/y extent of every rendered primitive, or ``None``."""
        xs, ys = [], []
        for shape in state.get("shapes", []):
            min_x, min_y, max_x, max_y = shape["bbox"]
            xs += [min_x, max_x]
            ys += [min_y, max_y]

        for seg in state.get("segments", []):
            xs += [seg[0], seg[2]]
            ys += [seg[1], seg[3]]

        for curve in state.get("curves", []):
            xs += [pt[0] for pt in curve]
            ys += [pt[1] for pt in curve]

        for pt in state.get("points", []):
            xs.append(pt[0])
            ys.append(pt[1])

        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    @classmethod
    def make_diagnostics_info(cls, state):
        """Return ``(intersections, degeneracies)`` as display-string rows."""
        diag = state.get("diagnostics", {})
        intersections = []

        for hit in diag.get("intersections", []):
            owner_a, owner_b = (cls._owner(sid) for sid in hit["shapes"])
            x, y = hit["point"]
            intersections.append(
                f"{owner_a} crosses {owner_b} at ({x:.4g}, {y:.4g})")
        degeneracies = []

        for deg in diag.get("degeneracies", []):
            degeneracies.append(
                f"{cls._owner(deg['shape'])}: {deg['type']} "
                f"({deg['reason']})")
        return intersections, degeneracies

    @staticmethod
    def _owner(shape_id):
        return "bare" if shape_id == -1 else f"shape {shape_id}"

    def set_world(self, world):
        """Cache ``world`` as the live source and render the chosen level."""
        self._world = world
        self._render()

    def _level(self):
        """Return the selected describe_state level."""
        for level, button in self._levels.items():
            if button.isChecked():
                return level
        return self.LEVELS[-1]

    def _on_level_toggled(self, checked):
        """Re-render once when the selected level changes."""
        if checked:
            self._render()

    def _describe(self, world, level):
        """Return the cached ``describe_state`` JSON for ``level``."""
        fingerprint = world.describe_state(level="basic")
        if fingerprint != self._fingerprint:
            self._fingerprint = fingerprint
            self._cache = {"basic": fingerprint}
        if level not in self._cache:
            self._cache[level] = world.describe_state(level=level)
        return self._cache[level]

    def _render(self):
        """Rebuild the tree from the held world at the selected level."""
        if self._world is None:
            key = None
        else:
            key = self._describe(self._world, self._level())
        if key == self._rendered_key:
            return
        self._rendered_key = key
        self._tree.clear()
        if key is None:
            QTreeWidgetItem(self._tree, ["No world loaded"])
            return
        state = json.loads(key)
        root = QTreeWidgetItem(self._tree, ["World (2D)"])
        for section, rows in self.make_world_info(state):
            group = QTreeWidgetItem(root, [section])
            for prop, value in rows:
                QTreeWidgetItem(group, [f"{prop}: {value}"])
            group.setExpanded(True)

        if "diagnostics" in state:
            self._add_diagnostics(root, state)

        root.setExpanded(True)
        # Widen the column so long entries are not clipped.
        self._tree.resizeColumnToContents(0)

    def _add_diagnostics(self, root, state):
        """Add the crossings and degeneracies, each as a counted subgroup."""
        intersections, degeneracies = self.make_diagnostics_info(state)
        group = QTreeWidgetItem(root, ["Diagnostics"])
        for label, rows in (("intersections", intersections),
                            ("degeneracies", degeneracies)):
            sub = QTreeWidgetItem(group, [f"{label}: {len(rows)}"])
            for row in rows:
                QTreeWidgetItem(sub, [row])
            sub.setExpanded(True)
        group.setExpanded(True)


class EntityTreePanel(_gui_common.PilotFeature):
    """Entity tree panel, toggled from the View "Panels" submenu."""

    # Poll period in milliseconds while the panel is visible. The canvas
    # mutates the world in C++ without a change signal, so the panel re-reads
    # it on this cadence; set_world makes an unchanged read a no-op.
    _POLL_MS = 500

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None
        self._timer = None

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Entity Tree", "Toggle the entity tree panel",
            None, id="panel.entity_tree", weight=30, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        """Show or hide the panel."""
        if checked:
            self._ensure_panel()
            self._refresh()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        """Build the dock lazily; poll it and follow sub-window changes."""
        if self._panel is not None:
            return
        self._panel = EntityTreeWidget()
        self._dock = QDockWidget("entity tree")
        self._dock.setWidget(self._panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           self._dock)
        # Keep the menu check in sync when the dock is closed by its button.
        self._dock.visibilityChanged.connect(self._action.setChecked)
        # Poll the world only while the panel is on screen.
        self._timer = QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self._refresh)
        self._dock.visibilityChanged.connect(self._on_visibility_changed)
        mdi = self._mdi_area()
        if mdi is not None:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)

    def _on_visibility_changed(self, visible):
        """Run the poll timer only while the panel is on screen."""
        if self._timer is None:
            return
        if visible:
            self._timer.start()
        else:
            self._timer.stop()

    def _on_subwindow_activated(self, _subwin):
        """Refresh the panel when the active sub-window changes."""
        if self._dock is not None and self._action.isChecked():
            QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        """Show the active sub-window's world."""
        self._panel.set_world(self._active_world())

    def _mdi_area(self):
        return self._mainWindow.centralWidget()

    def _active_world(self):
        """Return the active 2D viewer's world, or ``None``."""
        widget = self._mgr.currentR2DWidget()
        return None if widget is None else widget.world

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
