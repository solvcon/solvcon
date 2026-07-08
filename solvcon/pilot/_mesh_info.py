# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Dock panel showing StaticMesh geometry counts in a foldable tree."""

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDockWidget, QTreeWidgetItem

from .. import core
from . import _gui_common
from . import _mesh
from . import _tree_panel

__all__ = [  # noqa: F822
    'MeshInfo',
]


class MeshInfoTree(_tree_panel.TreePanelBase):
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


class MeshInfo(_gui_common.PilotFeature):
    """Mesh information panel, toggled from the View "Panels" submenu.

    The toggle item is placed under the "View/Panels" path. When on, the
    panel shows the active sub-window's mesh and follows the active
    sub-window; sub-windows without a mesh show "No mesh loaded".
    """

    def __init__(self, *args, **kw):
        self._status = kw.pop('style_status')
        super().__init__(*args, **kw)
        self._action = None
        self._dock = None
        self._panel = None

    def populate_menu(self):
        self._action = self.add_action(
            "View/Panels", "Mesh", "Toggle the mesh information panel", None,
            id="panel.mesh_info", weight=10, checkable=True)
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
        """Build the dock lazily and follow sub-window activation."""
        if self._panel is not None:
            return
        self._panel = MeshInfoTree(self._status)
        self._panel.boundary_toggled = self._on_boundary_toggled
        self._panel.edges_toggled = self._on_edges_toggled
        self._panel.normals_toggled = self._on_normals_toggled
        self._dock = QDockWidget("mesh")
        self._dock.setWidget(self._panel)
        self._mgr.mainWindow.addDockWidget(Qt.LeftDockWidgetArea,
                                           self._dock)
        # Keep the menu check in sync when the dock is closed by its button.
        self._dock.visibilityChanged.connect(self._action.setChecked)
        mdi = self._mdi_area()
        if mdi is not None:
            mdi.subWindowActivated.connect(self._on_subwindow_activated)

    def _on_subwindow_activated(self, _subwin):
        """Refresh the panel when the active sub-window changes.

        The refresh is deferred to the next event-loop pass because loading
        a mesh from a menu activates the new sub-window before ``updateMesh``
        populates it.
        """
        if self._dock is not None and self._action.isChecked():
            QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        """Show the active sub-window's mesh. The style boxes read the shared
        status, so rebuilding the tree already reflects the active viewer."""
        self._panel.set_mesh(self._active_mesh())

    def _on_boundary_toggled(self, ibc, checked):
        """Highlight or clear boundary set ``ibc`` in the active viewer."""
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showBoundary(ibc, checked)

    def _on_edges_toggled(self, checked):
        """Show or hide the feature-edge overlay in the active viewer."""
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showFeatureEdges(checked)

    def _on_normals_toggled(self, checked):
        """Show or hide the face-normal arrows in the active viewer."""
        widget = self._mgr.currentR3DWidget()
        if widget is not None:
            widget.showNormals(checked)

    def _mdi_area(self):
        return self._mainWindow.centralWidget()

    def _active_mesh(self):
        """Return the active 3D viewer's mesh, or ``None``.

        ``RManager.currentR3DWidget`` is used so the pybind11 widget that
        exposes ``mesh`` is reached; ``QMdiSubWindow.widget()`` would return
        a bare ``QWidget``.
        """
        widget = self._mgr.currentR3DWidget()
        return None if widget is None else widget.mesh

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
