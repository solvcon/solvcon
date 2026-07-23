# Copyright (c) 2021, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Show meshes.
"""

import os

from PySide6 import QtCore, QtGui, QtWidgets

from ... import core
from .. import _gui_common

__all__ = [  # noqa: F822
    'SampleMesh',
    'SampleMeshFeature',
    'MeshStyleStatus',
    'GmshFileDialog',
]


class SampleMesh:
    """
    The built-in sample meshes as data.

    Each ``mesh_*`` method builds and returns a :class:`StaticMesh` with no
    Qt, so the same example meshes serve the GUI feature and the console.
    """

    def make(self, name):
        """Build and return the sample mesh named ``name``, e.g. 'tetrahedron'
        for :meth:`mesh_tetrahedron`."""
        return getattr(self, 'mesh_' + name)()

    @classmethod
    def names(cls):
        """The sample-mesh names, in definition (menu) order."""
        return tuple(name[len('mesh_'):] for name in vars(cls)
                     if name.startswith('mesh_'))

    def mesh_triangle(self):
        mh = core.StaticMesh(ndim=2, nnode=4, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = (0, 0), (-1, -1), (1, -1), (0, 1)
        mh.cltpn.ndarray[:] = core.StaticMesh.TRIANGLE
        mh.clnds.ndarray[:, :4] = (3, 0, 1, 2), (3, 0, 2, 3), (3, 0, 3, 1)
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def mesh_tetrahedron(self):
        mh = core.StaticMesh(ndim=3, nnode=4, nface=4, ncell=1)
        mh.ndcrd.ndarray[:, :] = (0, 0, 0), (0, 1, 0), (-1, 1, 0), (0, 1, 1)
        mh.cltpn.ndarray[:] = core.StaticMesh.TETRAHEDRON
        mh.clnds.ndarray[:, :5] = [(4, 0, 1, 2, 3)]
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def mesh_solvcon_2dtext(self):
        Q = core.StaticMesh.QUADRILATERAL
        mh = core.StaticMesh(ndim=2, nnode=140, nface=0, ncell=65)
        mh.ndcrd.ndarray[:, :] = [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
            (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0),
            (14, 0), (15, 0), (16, 0), (18, 0), (19, 0), (20, 0), (21, 0),
            (22, 0), (23, 0), (24, 0), (25, 0), (26, 0), (27, 0), (28, 0),
            (29, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
            (7, 1), (8, 1), (9, 1), (10, 1), (11, 1), (12, 1), (13, 1),
            (14, 1), (15, 1), (16, 1), (18, 1), (19, 1), (20, 1), (21, 1),
            (22, 1), (23, 1), (24, 1), (25, 1), (26, 1), (27, 1), (28, 1),
            (29, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
            (7, 2), (8, 2), (9, 2), (12, 2), (13, 2), (15, 2), (16, 2),
            (18, 2), (19, 2), (22, 2), (23, 2), (24, 2), (25, 2), (26, 2),
            (27, 2), (28, 2), (29, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
            (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (11, 3), (12, 3), (13, 3),
            (15, 3), (16, 3), (17, 3), (18, 3), (19, 3), (20, 3), (21, 3),
            (22, 3), (23, 3), (24, 3), (25, 3), (26, 3), (27, 3), (28, 3),
            (29, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
            (7, 4), (8, 4), (9, 4), (11, 4), (12, 4), (13, 4), (15, 4),
            (16, 4), (17, 4), (18, 4), (19, 4), (20, 4), (21, 4), (22, 4),
            (23, 4), (24, 4), (25, 4), (26, 4), (27, 4), (0, 5), (1, 5),
            (2, 5), (3, 5)
        ]
        mh.cltpn.ndarray[:] = [
            Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,
            Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 0-20
            Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 21-31
            Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 32-45
            Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q, Q,  # 46-61
            Q, Q, Q  # 62-64
        ]
        mh.clnds.ndarray[:, :5] = [
            (4, 0, 1, 30, 29),
            (4, 1, 2, 31, 30),
            (4, 2, 3, 32, 31),
            (4, 4, 5, 34, 33),
            (4, 5, 6, 35, 34),
            (4, 6, 7, 36, 35),
            (4, 8, 9, 38, 37),
            (4, 9, 10, 39, 38),
            (4, 10, 11, 40, 39),
            (4, 12, 13, 42, 41),
            (4, 13, 14, 43, 42),
            (4, 14, 15, 44, 43),
            (4, 15, 16, 45, 44),
            (4, 17, 18, 47, 46),
            (4, 18, 19, 48, 47),
            (4, 19, 20, 49, 48),
            (4, 21, 22, 51, 50),
            (4, 22, 23, 52, 51),
            (4, 23, 24, 53, 52),
            (4, 25, 26, 55, 54),
            (4, 27, 28, 57, 56),
            (4, 31, 32, 61, 60),
            (4, 33, 34, 63, 62),
            (4, 35, 36, 65, 64),
            (4, 37, 38, 67, 66),
            (4, 41, 42, 69, 68),
            (4, 44, 45, 71, 70),
            (4, 46, 47, 73, 72),
            (4, 50, 51, 75, 74),
            (4, 52, 53, 77, 76),
            (4, 54, 55, 79, 78),
            (4, 56, 57, 81, 80),
            (4, 58, 59, 83, 82),
            (4, 59, 60, 84, 83),
            (4, 60, 61, 85, 84),
            (4, 62, 63, 87, 86),
            (4, 64, 65, 89, 88),
            (4, 66, 67, 91, 90),
            (4, 68, 69, 94, 93),
            (4, 70, 71, 96, 95),
            (4, 72, 73, 99, 98),
            (4, 74, 75, 103, 102),
            (4, 76, 77, 105, 104),
            (4, 78, 79, 107, 106),
            (4, 79, 80, 108, 107),
            (4, 80, 81, 109, 108),
            (4, 82, 83, 111, 110),
            (4, 86, 87, 115, 114),
            (4, 87, 88, 116, 115),
            (4, 88, 89, 117, 116),
            (4, 90, 91, 119, 118),
            (4, 92, 93, 121, 120),
            (4, 93, 94, 122, 121),
            (4, 95, 96, 124, 123),
            (4, 96, 97, 125, 124),
            (4, 98, 99, 127, 126),
            (4, 99, 100, 128, 127),
            (4, 100, 101, 129, 128),
            (4, 102, 103, 131, 130),
            (4, 103, 104, 132, 131),
            (4, 104, 105, 133, 132),
            (4, 106, 107, 135, 134),
            (4, 110, 111, 137, 136),
            (4, 111, 112, 138, 137),
            (4, 112, 113, 139, 138)
        ]
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def mesh_2dmix_small(self):
        T = core.StaticMesh.TRIANGLE
        Q = core.StaticMesh.QUADRILATERAL

        mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
        mh.ndcrd.ndarray[:, :] = [
            (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)
        ]
        mh.cltpn.ndarray[:] = [
            T, T, Q,
        ]
        mh.clnds.ndarray[:, :5] = [
            (3, 0, 3, 2, -1), (3, 0, 1, 3, -1), (4, 1, 4, 5, 3),
        ]
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def mesh_2dmix_large(self):
        T = core.StaticMesh.TRIANGLE
        Q = core.StaticMesh.QUADRILATERAL

        mh = core.StaticMesh(ndim=2, nnode=16, nface=0, ncell=14)
        mh.ndcrd.ndarray[:, :] = [
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (1, 1), (2, 1), (3, 1),
            (0, 2), (1, 2), (2, 2), (3, 2),
            (0, 3), (1, 3), (2, 3), (3, 3),
        ]
        mh.cltpn.ndarray[:] = [
            T, T, T, T, T, T,  # 0-5,
            Q, Q,  # 6-7
            T, T, T, T,  # 8-11
            Q, Q,  # 12-13
        ]
        mh.clnds.ndarray[:, :5] = [
            (3, 0, 5, 4, -1), (3, 0, 1, 5, -1),  # 0-1 triangles
            (3, 1, 2, 5, -1), (3, 2, 6, 5, -1),  # 2-3 triangles
            (3, 2, 7, 6, -1), (3, 2, 3, 7, -1),  # 4-5 triangles
            (4, 4, 5, 9, 8), (4, 5, 6, 10, 9),  # 6-7 quadrilaterals
            (3, 6, 7, 10, -1), (3, 7, 11, 10, -1),  # 8-9 triangles
            (3, 8, 9, 12, -1), (3, 9, 13, 12, -1),  # 10-11 triangles
            (4, 9, 10, 14, 13), (4, 10, 11, 15, 14),  # 12-13 quadrilaterals
        ]
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh

    def mesh_3dmix(self):
        HEX = core.StaticMesh.HEXAHEDRON
        TET = core.StaticMesh.TETRAHEDRON
        PSM = core.StaticMesh.PRISM
        PYR = core.StaticMesh.PYRAMID

        mh = core.StaticMesh(ndim=3, nnode=11, nface=0, ncell=4)
        mh.ndcrd.ndarray[:, :] = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
            (0.5, 1.5, 0.5),
            (1.5, 1, 0.5), (1.5, 0, 0.5),
        ]
        mh.cltpn.ndarray[:] = [
            HEX, PYR, TET, PSM,
        ]
        mh.clnds.ndarray[:, :9] = [
            (8, 0, 1, 2, 3, 4, 5, 6, 7), (5, 2, 3, 7, 6, 8, -1, -1, -1),
            (4, 2, 6, 9, 8, -1, -1, -1, -1), (6, 2, 6, 9, 1, 5, 10, -1, -1),
        ]
        mh.build_interior()
        mh.build_boundary()
        mh.build_ghost()
        return mh


class SampleMeshFeature(_gui_common.PilotFeature):
    """
    Create sample mesh windows from the built-in example meshes.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._samples = SampleMesh()

    def mesh_sample_dialog_entries(self):
        """``(category, label, tip, func)`` for each sample this feature can
        create; consumed by :class:`SampleMeshDialog`."""
        return [
            ("Basic shapes", "Triangle (2D)",
             "Create a very simple sample mesh of a triangle",
             self.mesh_triangle),
            ("Basic shapes", "Tetrahedron (3D)",
             "Create a very simple sample mesh of a tetrahedron",
             self.mesh_tetrahedron),
            ("Basic shapes", "\"solvcon\" text (2D)",
             "Create a sample mesh drawing a text string of \"solvcon\"",
             self.mesh_solvcon_2dtext),
            ("Mixed elements", "Small (2D)",
             "Create a small sample mesh of mixed elements in 2D",
             self.mesh_2dmix_small),
            ("Mixed elements", "Larger (2D)",
             "Create a larger simple sample mesh of mixed elements in 2D",
             self.mesh_2dmix_large),
            ("Mixed elements", "3D",
             "Create a very simple sample mesh of mixed elements in 3D",
             self.mesh_3dmix),
        ]

    def _show(self, name):
        """Build the sample mesh named ``name`` and open it in a fresh 3D
        viewer, echoing its edge count to the console."""
        mh = self._samples.make(name)
        w = self._mgr.add3DWidget()
        w.updateMesh(mh)
        w.showAxis(True)
        self._pycon.writeToHistory(f"{name} nedge: {mh.nedge}\n")

    def mesh_triangle(self):
        self._show('triangle')

    def mesh_tetrahedron(self):
        self._show('tetrahedron')

    def mesh_solvcon_2dtext(self):
        self._show('solvcon_2dtext')

    def mesh_2dmix_small(self):
        self._show('2dmix_small')

    def mesh_2dmix_large(self):
        self._show('2dmix_large')

    def mesh_3dmix(self):
        self._show('3dmix')


class SampleMeshDialog(_gui_common.PilotFeature):
    """A single Mesh-menu item that opens a dialog listing the example
    meshes, grouped by category, and creates the selected one.

    The example meshes are otherwise the bulk of the Mesh menu; collecting
    them behind one dialog keeps the menu itself for real mesh operations.
    Each contributing feature supplies its own
    ``mesh_sample_dialog_entries()`` and the controller passes the combined
    list in as ``entries``.
    """

    def __init__(self, *args, **kw):
        self._entries = kw.pop('entries', [])
        super().__init__(*args, **kw)
        self._dialog = None
        self._tree = None

    def populate_menu(self):
        self.add_action(
            "Mesh",
            text="Sample mesh dialog",
            tip="Choose an example mesh to create",
            func=self.open_dialog,
            id="mesh.sample_dialog",
            weight=10,
        )

    def open_dialog(self):
        # Build lazily and reuse: the dialog is modeless so several samples
        # can be created without reopening it.
        if self._dialog is None:
            self._dialog = self._build_dialog()
        self._dialog.show()
        self._dialog.raise_()

    def _build_dialog(self):
        dialog = QtWidgets.QDialog(self._mainWindow)
        dialog.setWindowTitle("Sample meshes")
        layout = QtWidgets.QVBoxLayout(dialog)

        tree = QtWidgets.QTreeWidget()
        tree.setHeaderHidden(True)
        groups = {}
        for category, label, tip, func in self._entries:
            group = groups.get(category)
            if group is None:
                group = QtWidgets.QTreeWidgetItem(tree, [category])
                group.setFlags(QtCore.Qt.ItemIsEnabled)
                group.setExpanded(True)
                groups[category] = group
            item = QtWidgets.QTreeWidgetItem(group, [label])
            item.setToolTip(0, tip)
            item.setData(0, QtCore.Qt.UserRole, func)
        tree.itemDoubleClicked.connect(
            lambda item, _col: self._invoke(item))
        layout.addWidget(tree)
        self._tree = tree

        buttons = QtWidgets.QHBoxLayout()
        create = QtWidgets.QPushButton("Create")
        create.setDefault(True)
        create.clicked.connect(lambda: self._invoke(tree.currentItem()))
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(dialog.close)
        buttons.addStretch(1)
        buttons.addWidget(create)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        return dialog

    def _invoke(self, item):
        """Create the mesh for ``item`` when it is a selectable leaf."""
        if item is None:
            return
        func = item.data(0, QtCore.Qt.UserRole)
        if callable(func):
            func()


class MeshStyleStatus(QtCore.QObject):
    """Shared on/off state of the three mesh styles for the active viewer.
    """

    # (style name understood by RDomainWidget, human-readable label), in one
    # canonical order shared by every mesh-style UI.
    STYLES = (
        ("surface", "Surface (lit shaded)"),
        ("wireframe", "Wireframe"),
        ("points", "Points"),
    )
    # The viewer default (wireframe only), reported when no viewer is active.
    _DEFAULTS = {"surface": False, "wireframe": True, "points": False}

    changed = QtCore.Signal()

    def __init__(self, *args, **kw):
        self._mgr = kw.pop('mgr')
        super().__init__(*args, **kw)
        self._actions = {}
        mdi = self._mgr.mainWindow.centralWidget()
        if mdi is not None:
            # A newly activated viewer brings its own styles; refresh the UIs.
            mdi.subWindowActivated.connect(lambda _sub: self.changed.emit())

    def is_shown(self, name):
        """Whether ``name`` is drawn in the active viewer, or its default."""
        widget = self._mgr.currentR3DWidget()
        if widget is None:
            return self._DEFAULTS.get(name, False)
        return widget.meshStyleShown(name)

    def set_shown(self, name, shown):
        """Show or hide ``name`` in the active viewer, then refresh the UIs."""
        widget = self._mgr.currentR3DWidget()
        if widget is None:
            self._mgr.pycon.writeToHistory(
                "No active 3D viewer to toggle the mesh style\n")
        else:
            widget.showMeshStyle(name, shown)
        self.changed.emit()

    def populate_menu(self):
        """Add the View > Mesh styles submenu of independent check items."""
        window = self._mgr.mainWindow
        submenu = self._mgr.menu_model.menu("View/Mesh styles")
        for name, label in self.STYLES:
            act = QtGui.QAction(label, window)
            act.setCheckable(True)
            act.setChecked(self.is_shown(name))
            act.setStatusTip(f"Show or hide the {label.lower()} mesh style")
            act.toggled.connect(
                lambda checked, n=name: self.set_shown(n, checked))
            self._actions[name] = act
            submenu.addAction(act)
        self.changed.connect(self._sync_menu)

    def _sync_menu(self):
        """Match the menu check marks to the active viewer's styles."""
        for name, act in self._actions.items():
            shown = self.is_shown(name)
            if act.isChecked() != shown:
                blocked = act.blockSignals(True)
                act.setChecked(shown)
                act.blockSignals(blocked)


class GmshFileDialog(_gui_common.PilotFeature):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._diag = QtWidgets.QFileDialog()
        self._diag.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        self._diag.setDirectory(self._get_initial_path())
        self._diag.setWindowTitle('Open Gmsh file ...')

    def run(self):
        self._diag.open(self, QtCore.SLOT('on_finished()'))

    def populate_menu(self):
        self.add_action(
            "File",
            text="Open Gmsh file",
            tip="Open Gmsh file",
            func=self.run,
            id="file.gmsh",
            weight=10,
        )

    @QtCore.Slot()
    def on_finished(self):
        filenames = []
        for path in self._diag.selectedFiles():
            filenames.append(path)
        self._load_gmsh_file(filename=filenames[0])

    @staticmethod
    def _get_initial_path():
        """
        Search for `tests/data/rectangle.msh` and return the directory holding
        it.  If not found, return an empty string.

        :return: The holding directory in absolute path or empty string.
        """
        found = ''
        for dp in ('.', core.__file__):
            dp = os.path.dirname(os.path.abspath(dp))
            dp2 = os.path.dirname(dp)
            while dp != dp2:
                tp = os.path.join(dp, "tests", "data")
                fp = os.path.join(tp, "rectangle.msh")
                if os.path.exists(fp):
                    found = tp
                    break
                dp = dp2
                dp2 = os.path.dirname(dp)
            if found:
                break
        return found

    def _load_gmsh_file(self, filename):
        if not os.path.exists(filename):
            self._pycon.writeToHistory(f"{filename} does not exist\n")
            return

        with open(filename, 'rb') as fobj:
            data = fobj.read()
        self._pycon.writeToHistory(f"gmsh mesh file {filename} is read\n")
        gmsh = core.Gmsh(data)
        mh = gmsh.to_block()
        self._pycon.writeToHistory("StaticMesh object created from gmsh\n")
        # Open a sub window for triangles and quadrilaterals:
        w = self._mgr.add3DWidget()
        w.updateMesh(mh)
        w.showAxis(True)
        self._pycon.writeToHistory(f"nedge: {mh.nedge}\n")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
