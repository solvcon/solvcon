# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The pilot inspector trees (mesh info and world entities) and the unified
dock that stacks them."""

import json

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget,
                               QTreeWidgetItem, QFrame, QDockWidget,
                               QStackedWidget, QHBoxLayout, QButtonGroup,
                               QRadioButton)

from .. import core
from . import _gui_common
from . import _mesh

__all__ = [  # noqa: F822
    'TreePanelBase',
    'MeshInfoTree',
    'EntityTreeWidget',
    'TreePanel',
]


class TreePanelBase(QWidget):
    """Base widget wrapping a single-column tree for a dock panel.

    The base owns the tree widget, the frameless single-column look, and
    the helpers that render ``(section, rows)`` groups and a placeholder
    row. A subclass fills the tree from its own source and may add header
    controls above the tree through :meth:`_build_header`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tree = QTreeWidget()
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        # Drop the tree frame so its scroll bar sits flush in the panel.
        self._tree.setFrameShape(QFrame.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._build_header(layout)
        layout.addWidget(self._tree)
        self.setLayout(layout)

    def _build_header(self, layout):
        """Add controls above the tree; the base panel adds none."""

    def _show_placeholder(self, text):
        """Clear the tree and show a single ``text`` row."""
        self._tree.clear()
        QTreeWidgetItem(self._tree, [text])

    def _render_sections(self, parent, sections):
        """Render ``(section, rows)`` groups under ``parent``.

        Each section is a foldable node holding ``prop: value`` rows.
        """
        for section, rows in sections:
            group = QTreeWidgetItem(parent, [section])
            for prop, value in rows:
                QTreeWidgetItem(group, [f"{prop}: {value}"])
            group.setExpanded(True)

    def _finalize_root(self, root):
        """Expand ``root`` and widen the column to fit the contents."""
        root.setExpanded(True)
        self._tree.resizeColumnToContents(0)


class MeshInfoTree(TreePanelBase):
    """Widget that presents the mesh information tree inside the dock.

    :ivar boundary_toggled:
        Owner-supplied callback ``boundary_toggled(ibc, checked)`` that routes
        a boundary set's check-box to the active viewer; ``None`` until wired.
    :vartype boundary_toggled: callable or None
    """

    # Data roles that route a tree item's check-box toggle; the boundary
    # index and the style name ride in the roles after the kind.
    _ROLE_KIND = Qt.UserRole
    _ROLE_IBC = Qt.UserRole + 1
    _ROLE_STYLE = Qt.UserRole + 2

    # Map cell type numbers to human-readable names.
    CELL_TYPE_NAME = {
        core.StaticMesh.POINT: "point",
        core.StaticMesh.LINE: "line",
        core.StaticMesh.QUADRILATERAL: "quadrilateral",
        core.StaticMesh.TRIANGLE: "triangle",
        core.StaticMesh.HEXAHEDRON: "hexahedron",
        core.StaticMesh.TETRAHEDRON: "tetrahedron",
        core.StaticMesh.PRISM: "prism",
        core.StaticMesh.PYRAMID: "pyramid",
    }

    def __init__(self, style_status=None, mh=None, parent=None):
        super().__init__(parent)
        self.style_status = style_status
        if self.style_status is not None:
            self.style_status.changed.connect(self.refresh_style_checks)
        self._style_items = {}
        self.boundary_toggled = None
        self.edges_toggled = None
        self.normals_toggled = None
        self._building = False
        self._tree.itemChanged.connect(self._on_item_changed)
        self.set_mesh(mh)

    @classmethod
    def make_mesh_info(cls, mh):
        """Build the mesh information as ``(section, rows)`` groups.

        Each group pairs a heading with its ``[property, value]`` string
        rows, so the panel renders one foldable tree node per group.
        """
        sections = [
            ("Counts", [
                ["dim", str(mh.ndim)],
                ["node", str(mh.nnode)],
                ["face", str(mh.nface)],
                ["cell", str(mh.ncell)],
                ["edge", str(mh.nedge)],
                ["bound", str(mh.nbound)],
                ["bcs", str(mh.nbcs)],
            ]),
            ("Ghost", [
                ["node", str(mh.ngstnode)],
                ["face", str(mh.ngstface)],
                ["cell", str(mh.ngstcell)],
            ]),
        ]
        # Ghost entities are stored first; measure only the body entities.
        crd = mh.ndcrd.ndarray[mh.ndcrd.nghost:]
        if crd.size:
            lower = crd.min(axis=0)
            upper = crd.max(axis=0)
            bbox = [[axis, f"[{lower[it]:.4g}, {upper[it]:.4g}]"]
                    for it, axis in zip(range(mh.ndim), "xyz")]
            sections.append(("Bounding box", bbox))
        # Tally the cell types over the body (non-ghost) cells.
        tpn = mh.cltpn.ndarray[mh.cltpn.nghost:]
        cells = []
        for tnum, name in sorted(cls.CELL_TYPE_NAME.items()):
            count = int(np.count_nonzero(tpn == tnum))
            if count:
                cells.append([name, str(count)])
        if cells:
            sections.append(("Cell types", cells))
        return sections

    @classmethod
    def make_boundary_info(cls, mh):
        """Return one ``[ibc, nface]`` row per boundary set.

        Each boundary face records its set index in column 1 of ``bndfcs``,
        so grouping the rows by that index yields the face count of every
        set, including the trailing catch-all set of unspecified faces.
        """
        bnd = mh.bndfcs.ndarray
        if bnd.size:
            counts = np.bincount(bnd[:, 1], minlength=mh.nbcs)
        else:
            counts = np.zeros(mh.nbcs, dtype='int64')
        return [[ibc, int(counts[ibc])] for ibc in range(mh.nbcs)]

    def set_mesh(self, mh):
        """Rebuild the tree from ``mh``, or show "No mesh loaded" when None."""
        self._building = True
        try:
            if mh is None:
                self._show_placeholder("No mesh loaded")
                return
            self._tree.clear()
            root = QTreeWidgetItem(self._tree, [f"StaticMesh ({mh.ndim}D)"])
            # Keep the display toggles (styles and overlays, then boundaries)
            # together at the top, above the read-only information sections.
            self._add_style_toggles(root)
            self._add_overlay_toggles(root)
            self._add_boundary_group(root, mh)
            self._render_sections(root, self.make_mesh_info(mh))
            self._finalize_root(root)
        finally:
            self._building = False

    def _add_style_toggles(self, root):
        """Add the mesh style on-off check boxes.

        Each mirrors the active viewer's style through the shared status, so a
        fresh viewer shows the wireframe checked and the other two clear.
        """
        self._style_items = {}
        if self.style_status is None:
            return
        for name, label in _mesh.MeshStyleStatus.STYLES:
            item = QTreeWidgetItem(root, [label])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(0, self._ROLE_KIND, 'style')
            item.setData(0, self._ROLE_STYLE, name)
            shown = self.style_status.is_shown(name)
            item.setCheckState(0, Qt.Checked if shown else Qt.Unchecked)
            self._style_items[name] = item

    def refresh_style_checks(self):
        """Match the style check boxes to the active viewer's styles."""
        if self.style_status is None:
            return
        self._building = True
        try:
            for name, item in self._style_items.items():
                shown = self.style_status.is_shown(name)
                item.setCheckState(0, Qt.Checked if shown else Qt.Unchecked)
        finally:
            self._building = False

    def _add_overlay_toggles(self, root):
        """Add the feature-edge and face-normal overlay check boxes.

        Both default off; each drives its own viewer overlay.
        """
        for label, kind in (("feature edges", 'edges'),
                            ("normals", 'normals')):
            item = QTreeWidgetItem(root, [label])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(0, self._ROLE_KIND, kind)
            item.setCheckState(0, Qt.Unchecked)

    def _add_boundary_group(self, root, mh):
        """Add the boundary sets as a group of check boxes (default off)."""
        binfo = self.make_boundary_info(mh)
        if not binfo:
            return
        group = QTreeWidgetItem(root, ["Boundaries"])
        for ibc, count in binfo:
            item = QTreeWidgetItem(group, [f"bc {ibc}: {count} faces"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setData(0, self._ROLE_KIND, 'boundary')
            item.setData(0, self._ROLE_IBC, ibc)
            item.setCheckState(0, Qt.Unchecked)
        group.setExpanded(True)

    def _on_item_changed(self, item, _column):
        """Route a check-box toggle to its handler, ignoring plain rows."""
        if self._building:
            return
        checked = item.checkState(0) == Qt.Checked
        kind = item.data(0, self._ROLE_KIND)
        if kind == 'boundary' and self.boundary_toggled is not None:
            self.boundary_toggled(item.data(0, self._ROLE_IBC), checked)
        elif kind == 'style' and self.style_status is not None:
            self.style_status.set_shown(
                item.data(0, self._ROLE_STYLE), checked)
        elif kind == 'edges' and self.edges_toggled is not None:
            self.edges_toggled(checked)
        elif kind == 'normals' and self.normals_toggled is not None:
            self.normals_toggled(checked)


class EntityTreeWidget(TreePanelBase):
    """Widget that presents the entity tree inside the dock."""

    LEVELS = ("basic", "diagnostics")

    # Poll period in milliseconds while the tree is on screen. The 2D canvas
    # mutates its world in C++ without a change signal, so the tree re-reads
    # the world it holds on this cadence; the describe_state cache makes an
    # unchanged tick a no-op.
    _POLL_MS = 500

    # A unique sentinel for "nothing rendered/fingerprinted yet", distinct
    # from the ``None`` key that marks the "No world loaded" render.
    _UNSET = object()

    def __init__(self, world=None, parent=None):
        self._levels = {}
        super().__init__(parent)
        self._world = None
        self._fingerprint = self._UNSET
        self._cache = {}
        self._rendered_key = self._UNSET
        self._timer = QTimer(self)
        self._timer.setInterval(self._POLL_MS)
        self._timer.timeout.connect(self._render)
        self.set_world(world)

    def showEvent(self, event):
        """Poll the held world only while the tree is on screen."""
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):
        """Stop polling once the tree leaves the screen."""
        super().hideEvent(event)
        self._timer.stop()

    def _build_header(self, layout):
        """Add the basic/diagnostics level selector above the tree."""
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
        layout.addLayout(level_row)

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
        if key is None:
            self._show_placeholder("No world loaded")
            return
        self._tree.clear()
        state = json.loads(key)
        root = QTreeWidgetItem(self._tree, ["World (2D)"])
        self._render_sections(root, self.make_world_info(state))

        if "diagnostics" in state:
            self._add_diagnostics(root, state)

        self._finalize_root(root)

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


class TreePanel(_gui_common.PilotFeature):
    """Unified inspector dock that follows the active sub-window.

    One dock holds both trees in a stack. A 3D mesh viewer shows the mesh
    information tree; the 2D canvas shows the world entity tree. The active
    sub-window's type selects which tree is shown, so the two panels that
    used to be separate now share one widget.
    """

    def __init__(self, *args, **kw):
        self._status = kw.pop('style_status')
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._stack = None
        self._mesh_tree = None
        self._entity_tree = None

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Inspector", "Toggle the inspector panel",
            None, id="panel.inspector", weight=10, checkable=True)
        self._action.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked):
        """Show or hide the panel."""
        if checked:
            self._ensure_panel()
            self._sync()
            self._dock.show()
        elif self._dock is not None:
            self._dock.hide()

    def _ensure_panel(self):
        """Build the dock lazily and follow sub-window activation."""
        if self._stack is not None:
            return
        self._mesh_tree = MeshInfoTree(self._status)
        self._mesh_tree.boundary_toggled = self._on_boundary_toggled
        self._mesh_tree.edges_toggled = self._on_edges_toggled
        self._mesh_tree.normals_toggled = self._on_normals_toggled
        self._entity_tree = EntityTreeWidget()
        self._stack = QStackedWidget()
        self._stack.addWidget(self._mesh_tree)
        self._stack.addWidget(self._entity_tree)
        self._dock = QDockWidget("inspector")
        self._dock.setWidget(self._stack)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           self._dock)
        self._dock.visibilityChanged.connect(self._action.setChecked)
        mdi = self._mdi_area()
        if mdi is not None:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)

    def _on_subwindow_activated(self, _subwin):
        """Re-select the tree when the active sub-window changes."""
        if self._dock is not None and self._action.isChecked():
            QTimer.singleShot(0, self._sync)

    def _sync(self):
        """Select the tree that matches the active sub-window.

        A 3D viewer shows the mesh tree; the 2D canvas shows the world
        tree. Detection keys on which viewer the manager reports active.
        """
        widget3d = self._mgr.currentR3DWidget()
        if widget3d is not None:
            self._stack.setCurrentWidget(self._mesh_tree)
            self._mesh_tree.set_mesh(widget3d.mesh)
            return
        widget2d = self._mgr.currentR2DWidget()
        if widget2d is not None:
            self._stack.setCurrentWidget(self._entity_tree)
            self._entity_tree.set_world(widget2d.world)

    def _on_boundary_toggled(self, ibc, checked):
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showBoundary(ibc, checked)

    def _on_edges_toggled(self, checked):
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showFeatureEdges(checked)

    def _on_normals_toggled(self, checked):
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showNormals(checked)

    def _mdi_area(self):
        return self._mainWindow.centralWidget()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
