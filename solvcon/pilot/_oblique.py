# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Example pilot apps for the oblique-shock reflection.

The mesh construction, boundary tagging, and solver driver live in
:mod:`solvcon.multidim.euler.oblique`.  :class:`ObliqueShockMesh` draws the
mesh in a 3D widget and reports the boundary classification (inlet / slip wall
/ outflow) to the console; :class:`ObliqueShockSolver` runs the Euler driver
and animates the evolving density as a flat 2D color field, drawn with the
native ``RDomainWidget.updateColorField`` per-cell-colored triangles.
"""

import numpy as np

from PySide6 import QtCore

from .. import core
from ..multidim.euler import oblique
from . import _gui_common

__all__ = [  # noqa: F822
    'ObliqueShockMesh',
    'ObliqueShockSolver',
]


def _colormap(t):
    """Map ``t`` in [0, 1] to a jet-like RGB array (``..., 3``) in [0, 1].

    A four-stop blue-cyan-yellow-red ramp; the triangle-wave channels are the
    standard compact "jet" approximation, enough to read a scalar field
    without pulling in a plotting dependency.
    """
    t = np.clip(np.asarray(t, dtype='float64'), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _cell_triangulation(mh):
    """
    Per-cell unshared triangle vertices and indices for a flat color field.
    """
    verts, tris, nnds = [], [], []
    base = 0
    for icl in range(mh.ncell):
        nnd = mh.clnds[icl, 0]
        nnds.append(nnd)
        for it in range(nnd):
            ind = mh.clnds[icl, 1 + it]
            verts.append((mh.ndcrd[ind, 0], mh.ndcrd[ind, 1], 0.0))
        for it in range(1, nnd - 1):
            tris.append((base, base + it, base + it + 1))
        base += nnd
    return (np.array(verts, dtype='float32'),
            np.array(tris, dtype='uint32'),
            np.array(nnds, dtype='int64'))


def _field_colors(field, nnds, vmin, vmax):
    span = vmax - vmin
    t = (field - vmin) / span if span > 0 else np.zeros_like(field)
    return np.repeat(_colormap(t), nnds, axis=0).astype('float32')


class ObliqueShockMesh(_gui_common.PilotFeature):
    """
    Draw the oblique-shock reflection mesh and tag its boundary.
    """

    def mesh_sample_dialog_entries(self):
        cat = "Oblique-shock reflection"
        return [
            (cat, "Quad mesh (2D)",
             "Draw the quad wedge mesh for the oblique-shock reflection",
             self.draw_quad_mesh),
            (cat, "Triangle mesh (2D)",
             "Draw the triangle wedge mesh for the oblique-shock reflection",
             self.draw_triangle_mesh),
            (cat, "Unstructured mesh (2D)",
             "Draw the unstructured (Delaunay) triangle wedge mesh for the "
             "oblique-shock reflection",
             self.draw_unstructured_mesh),
        ]

    def draw_quad_mesh(self):
        self._draw_mesh('quad')

    def draw_triangle_mesh(self):
        self._draw_mesh('triangle')

    def draw_unstructured_mesh(self):
        self._draw_mesh('unstructured')

    def _draw_mesh(self, cell_type):
        mesher = oblique.ObliqueShockMesher()
        mh = mesher.make_mesh(cell_type=cell_type)
        inlet, walls, outflow = mesher.classify_boundary(mh)
        w = self._mgr.add3DWidget()
        w.updateMesh(mh)
        w.showAxis(True)
        self._pycon.writeToHistory(
            f"oblique-shock {cell_type} mesh: {mh.ncell} cells, "
            f"{mh.nedge} edges\n"
            f"boundary faces: {len(inlet)} inlet, {len(walls)} slip wall, "
            f"{len(outflow)} outflow\n")


class ObliqueShockSolver(_gui_common.PilotFeature):
    """
    Run the oblique-shock Euler driver and animate the density field.
    """

    #: Solver steps marched per timer frame.
    STEPS_PER_FRAME = 5
    #: Stop the animation after this many steps.
    MAX_STEPS = 2000
    #: Qt timer interval in milliseconds.
    INTERVAL_MS = 50

    def __init__(self, *args, **kw):
        # Keep every running session (3D widget, timer, driver) referenced so
        # Qt and the driver are not garbage-collected mid-run.
        self._sessions = []
        super(ObliqueShockSolver, self).__init__(*args, **kw)

    def populate_menu(self):
        self.add_action(
            "Mesh",
            text="(WIP) Oblique-shock solution (density)",
            tip="March the oblique-shock Euler solver and draw the density "
                "as a 2D color field",
            func=self._run,
            id="mesh.oblique_solver",
            weight=20,
        )

    def _run(self):
        shock = oblique.ObliqueShock()
        shock.build_constant()
        shock.build_numerical(cell_type='quad')

        # The cell geometry is fixed across the run; only the colors change,
        # so triangulate once and cache the vertex/index arrays.
        verts, indices, nnds = _cell_triangulation(shock.mesh)
        widget = self._mgr.add3DWidget()
        widget.showAxis(True)
        timer = QtCore.QTimer()
        session = dict(shock=shock, widget=widget, timer=timer, nnds=nnds,
                       verts=core.SimpleArrayFloat32(array=verts),
                       indices=core.SimpleArrayUint32(array=indices), step=0)
        self._draw_frame(session)
        timer.timeout.connect(lambda: self._advance(session))
        timer.start(self.INTERVAL_MS)
        self._sessions.append(session)

    def _advance(self, session):
        if session['step'] >= self.MAX_STEPS:
            session['timer'].stop()
            return
        session['shock'].march(self.STEPS_PER_FRAME)
        session['step'] += self.STEPS_PER_FRAME
        self._draw_frame(session)

    def _draw_frame(self, session):
        svr = session['shock'].svr
        # Density is the first conserved variable; the raw .ndarray view
        # prepends the ghost rows, so colour only the body cells.
        field = svr.so0n.ndarray[svr.ngstcell:, 0]
        colors = _field_colors(field, session['nnds'],
                               field.min(), field.max())
        session['widget'].updateColorField(
            session['verts'], core.SimpleArrayFloat32(array=colors),
            session['indices'])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
